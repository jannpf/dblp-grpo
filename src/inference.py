import os
import re
import json
import time

import dotenv
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel, PeftConfig

from DBLPDataset import DblpDataset, DblpCollator


print("CUDA available:", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Current device:", torch.cuda.current_device())
torch.cuda.empty_cache()
print("Emptied cache")

dotenv.load_dotenv()
# OPENAI_API_URL = os.getenv("OPENAI_API_URL")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EXPERIMENT = os.getenv("EXPERIMENT")
DATASET_PATH = os.getenv("DATASET_PATH")
MODEL = os.getenv("MODEL")
ADAPTER = os.getenv("ADAPTER")

if None in (EXPERIMENT, DATASET_PATH, MODEL):
    raise ValueError("Settings not found. Check .env file")

DATASET_PATH = os.path.expanduser(DATASET_PATH)
MODEL = os.path.expanduser(MODEL)

if ADAPTER:
    ADAPTER = os.path.expanduser(ADAPTER)

N_SAMPLES = None  # all


def extract_answer(completion: str) -> str:
    """
    Returns the text after the *last* </think> tag, stripped.
    Falls back to the last <answer>...</answer> block if present.
    Also unwraps a final fenced code block (```sparql ... ```), if used.
    """

    # Legacy fallback: use the last <answer>...</answer>
    legacy = re.findall(r"<answer>(.*?)</answer>", completion, flags=re.DOTALL | re.IGNORECASE)
    if legacy:
        return legacy[-1].strip()

    # Prefer everything after the *last* </think> if present
    last_think_end = 0
    for m in re.finditer(r"</think\s*>", completion, flags=re.IGNORECASE):
        last_think_end = m.end()

    tail = completion[last_think_end:]

    # If there is a fenced code block in the tail, take the last fenced block
    code_blocks = re.findall(r"```(?:sparql|sql)?\s*(.*?)```", tail, flags=re.DOTALL | re.IGNORECASE)
    if code_blocks:
        return code_blocks[-1].strip()

    return tail.strip()


def log(msg):
    print(f"{time.strftime('%Y%m%d-%H%M%S')} - {msg}")


def load_model(model_path, adapter_path=None, device="auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        device_map=device
    )

    if adapter_path and os.path.isdir(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    return model, tokenizer


def main():
    start = time.perf_counter()

    results = dict()
    errors = []
    queries = []

    enable_cot = False

    system_prompt = """You are a SPARQL expert.
    Your task is to generate a syntactically and semantically correct SPARQL query that answers a given natural language question, using only the provided entities and relations.
    Output the final SPARQL query — without any extra explanation or formatting.

    Query Generation Rules:
    - Only use the provided entities and relations; do not invent or infer additional ones.
    - Use all provided entities and relations in the query.
    - Do not use prefixes; write all URIs in full.
    - By default, use SELECT DISTINCT in your queries, unless the context clearly requires otherwise.
    - For yes/no questions, use the ASK form.
    - Carefully consider whether the answer should be the subject or object in each relevant triple pattern.
    - Always use single quotes for literals
"""

    if enable_cot:
        system_prompt = """You are a SPARQL expert.
    Your task is to generate a syntactically and semantically correct SPARQL query that answers a given natural language question, using only the provided entities and relations.

    Please follow these instructions:
    - Think step by step, reasoning through the transformation from question to query.
    - Enclose your detailed reasoning in <think> ... </think> tags.
    - Output the final SPARQL query — without any extra explanation or formatting.

    Query Generation Rules:
    - Only use the provided entities and relations; do not invent or infer additional ones.
    - Use all provided entities and relations in the query.
    - Do not use prefixes; write all URIs in full.
    - By default, use SELECT DISTINCT in your queries, unless the context clearly requires otherwise.
    - For yes/no questions, use the ASK form.
    - Carefully consider whether the answer should be the subject or object in each relevant triple pattern.
    - Always use single quotes for literals

    Example output format:
    <think> Step-by-step reasoning here. </think> SPARQL query here
"""

    # exclude
    query_types = (
        'SUPERLATIVE+COMPARATIVE',
        'BOOLEAN',
        'SINGLE_FACT',
        'NEGATION',
        'DOUBLE_NEGATION',
        'UNION',
        'MULTI_FACT',
        'DISAMBIGUATION',
        'DOUBLE_INTENT',
        'COUNT',
    )

    model, tokenizer = load_model(MODEL, ADAPTER)
    model.eval()

    test_ds = DblpDataset(
        DATASET_PATH,
        subset="test",
        n=N_SAMPLES,
        resource_details=True,
        system_prompt=system_prompt,
        query_types=query_types,
    )

    collator = DblpCollator(
        tokenizer=tokenizer,
        max_length=700,
        enable_thinking=True if enable_cot else False,
    )

    test_data_loader = DataLoader(
        test_ds,
        collate_fn=collator,
        batch_size=32,
    )

    # set_global_model_client(llm_client)
    # print(f"Peft config: {llm_client.model.peft_config}")
    # print(f"Active Peft config: {llm_client.model.active_peft_config}")
    print(f"Model generate options: {model.generation_config}")

    log(f"Dataset shape: {test_ds.questions.shape}")
    with torch.inference_mode():
        for batch in tqdm(test_data_loader):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1024 if enable_cot else 384
            )

            # prompt_lengths = attention_mask.sum(dim=1)
            # completion_token_seqs = [
            #     generated_ids[i, prompt_lengths[i] :].tolist()
            #     for i in range(generated_ids.size(0))
            # ]

            T_in = input_ids.shape[1]
            gen_tokens = generated_ids[:, T_in:]

            texts = tokenizer.batch_decode(
                gen_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            for i in range(len(texts)):
                generated_full = texts[i]
                generated_query = extract_answer(generated_full)

                if generated_query:
                    queries.append({
                        "id": batch["ids"][i],
                        "query": generated_query,
                        "prompt": batch["prompts"][i],
                        "answer": generated_full,
                    })

    ittook = time.perf_counter() - start
    log(f"Inference for {len(test_data_loader)} samples completed in {ittook:.2f}")

    results["parameters"] = {
        "dataset": test_ds.data_dir,
        "samples": N_SAMPLES,
        "model": MODEL,
        "system_prompt": system_prompt,
        "generation_config": model.generation_config.to_json_string(ignore_metadata=True),
        "cot": enable_cot,
    }
    results["errors"] = errors
    results["queries"] = queries

    result_file_name = f"{EXPERIMENT.rstrip('/')}/results/{time.strftime('%Y%m%d-%H%M%S')}_results.json"
    log(f"Writing results to {result_file_name}")
    with open(result_file_name, "wt") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
