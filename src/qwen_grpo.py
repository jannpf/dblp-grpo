import os
import re
import time

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from datasets import Dataset as HFDataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from trl import GRPOTrainer, GRPOConfig
import torch
import numpy as np
import dotenv

from DBLPDataset import DblpDataset
from QueryCache import QueryCache
from sparql import execute_sparql, validate_query
from utils import tokenize_sparql, canonicalize_variables, remove_aliases, normalize_sparql, normalize_results, fbeta_score

print("CUDA available:", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
torch.cuda.empty_cache()
print("Emptied cache")

# os.environ["RANK"] = "0"
# os.environ["LOCAL_RANK"] = "0"
# os.environ["WORLD_SIZE"] = "1"
# os.environ['MASTER_ADDR'] = "localhost"
# os.environ['MASTER_PORT'] = "12355"

dotenv.load_dotenv()
SPARQL_ENDPOINT = os.getenv("SPARQL_ENDPOINT")
DATASET_PATH = os.getenv("DATASET_PATH")
BASE_MODEL = os.getenv("BASE_MODEL")
EXPERIMENT = os.getenv("EXPERIMENT")

if None in (DATASET_PATH, BASE_MODEL, EXPERIMENT):
    raise ValueError("Some Settings not found. Check .env file")

MODEL_OUTPUT = f"{EXPERIMENT.rstrip('/')}/checkpoints"
LOGGING_DIR = f"{EXPERIMENT.rstrip('/')}/logs"

DATASET_PATH = os.path.expanduser(DATASET_PATH)
BASE_MODEL = os.path.expanduser(BASE_MODEL)

N_SAMPLES = None  # 1200 # 64

# query result caching
CACHE_FILE = os.path.join(DATASET_PATH, "query_cache.pkl")
query_cache = QueryCache(max_size=8000, cache_file=CACHE_FILE)

WEIGHTS = [
    3,  # ef_reward
    # 2,  # similarity_reward
    1,  # structure_reward
    0.5,  # format_reward
    1,  # length_reward
    # 1,  # query_length_reward
]

SYSTEM_PROMPT = """You are a SPARQL expert.
    Your task is to generate a syntactically and semantically correct SPARQL query that answers a given natural language question, using only the provided entities and relations.

    Please follow these instructions:
    - carefully analyze the given question, pay special attention to negations (not, etc.).
    - Think step by step, reasoning through the transformation from question to query.
    - Enclose your detailed reasoning in <think> ... </think> tags.
    - Output the final SPARQL query — without any extra explanation or formatting.

    Query Generation Rules:
    - Only use the provided entities and relations; do not invent or infer additional ones.
    - Use all provided entities and relations in the query.
    - Do not use prefixes; write all URIs in full.
    - By default, use SELECT DISTINCT in your queries, unless the context clearly requires otherwise.
    - For yes/no questions, always use the ASK keyword to obtain a boolean result
    - Carefully consider whether the answer should be the subject or object in each relevant triple pattern.
    - Always use single quotes for literals

    Example output format:
    <think> Step-by-step reasoning here. </think> SPARQL query here
"""


def extract_answer(completion: str) -> str:
    """
    Returns the text after the *last* </think> tag, stripped.
    Falls back to the last <answer>...</answer> block if present.
    Also unwraps a final fenced code block (```sparql ... ```), if used.
    """

    # Legacy fallback: use the last <answer>...</answer>
    legacy = re.findall(
        r"<answer>(.*?)</answer>",
        completion,
        flags=re.DOTALL | re.IGNORECASE
    )

    if legacy:
        return legacy[-1].strip()

    # Prefer everything after the *last* </think> if present
    last_think_end = 0
    for m in re.finditer(r"</think\s*>", completion, flags=re.IGNORECASE):
        last_think_end = m.end()

    tail = completion[last_think_end:]

    # If there is a fenced code block in the tail, take the last fenced block
    code_blocks = re.findall(
        r"```(?:sparql|sql)?\s*(.*?)```",
        tail,
        flags=re.DOTALL | re.IGNORECASE
    )

    if code_blocks:
        return code_blocks[-1].strip()

    return tail.strip()


def log(msg):
    print(f"{time.strftime('%Y%m%d-%H%M%S')} - {msg}")


def process_results(results):
    answer_obj = dict()
    answer_obj["id"] = 0
    answer_obj["answer"] = results

    return DblpDataset.process_answers([answer_obj]).loc[0, "answer"]


# REWARD FUNCTIONS

def query_similarity(completion: str, query: str) -> float:
    completion_normalized = normalize_sparql(completion)
    query_normalized = normalize_sparql(query)

    completion_tokenized = tokenize_sparql(completion_normalized)
    query_tokenized = tokenize_sparql(query_normalized)

    return sentence_bleu(
        [query_tokenized],
        completion_tokenized,
        smoothing_function=SmoothingFunction().method3
    )


def execution_feedback(completion: str, answer: str) -> float:
    query_normalized = normalize_sparql(completion)
    cached_answer = query_cache.get(query_normalized)

    if cached_answer is not None:
        log(f"Cache hit for execution feedback")
        generated_answer = cached_answer
    else:
        try:
            query = completion.strip().lower()
            if query.startswith("select"):
                # add a limit clause, as no example in the test set has more than 3000 entries
                query = query.rstrip(" ;") + f"\nLIMIT 3000"
            elif query.startswith(("ask", "describe", "construct")):
                pass
            else:
                # invalid query
                log(f"Query starts with unknown keyword: {completion[:100]}")
                return -0.5
            results = execute_sparql(query, SPARQL_ENDPOINT)
        except Exception as e_limited:
            try:
                # try to run the original query
                # if this works, adding the limit caused the issue
                if query.startswith("select"):
                    results = execute_sparql(completion, SPARQL_ENDPOINT)
                    log(f"WARNING: the limit clause broke the query: {query}")
                else:
                    raise e_limited
            except Exception as e_original:
                log(f"Exception with original query: {completion}: {e_original}")
                return -0.5

        generated_answer = process_results(results)

        # the torch dataloader converts tuples to lists
        def tuples_to_lists(obj):
            if isinstance(obj, tuple):
                return [tuples_to_lists(item) for item in obj]
            elif isinstance(obj, list):
                return [tuples_to_lists(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: tuples_to_lists(v) for k, v in obj.items()}
            else:
                return obj

        generated_answer = tuples_to_lists(generated_answer)
        query_cache.put(query_normalized, generated_answer)

    results_normalized = normalize_results(generated_answer)
    reference_normalized = normalize_results(answer)

    # exact match
    if results_normalized == reference_normalized:
        return 1.0

    return fbeta_score(
        results_normalized,
        reference_normalized,
        beta=1.0,
    )


def contains_entities(completion: list, entities: list) -> float:
    if all([e in completion for e in entities]):
        return 1
    return 0


def contains_relations(completion: list, relations: list) -> float:
    if all([r in completion for r in relations]):
        return 1
    return 0


def query_length(completion: str, query: str) -> float:
    completion_normalized = normalize_sparql(completion)
    query_normalized = normalize_sparql(query)

    completion_tokenized = tokenize_sparql(completion_normalized)
    query_tokenized = tokenize_sparql(query_normalized)

    completion_len = len(completion_tokenized)
    query_len = len(query_tokenized)

    # length ratio
    r = completion_len / query_len

    # symmetric logarithmic distance
    # empiric value
    alpha = 2

    return np.exp(-alpha * np.abs(np.log(r)))


def structure_reward(prompts, completions, entities, relations, **kwargs):
    rewards = []

    for i in range(len(prompts)):
        # expect conversational format
        assert completions[i][-1]["role"] == "assistant", ValueError(
            "Unexpected message: {}".format(completions)
        )

        completion = completions[i][-1]["content"]
        generated_query = extract_answer(completion)

        reward = 0

        reward += 0.5 * contains_relations(generated_query, relations[i])
        reward += 0.5 * contains_entities(generated_query, entities[i])

        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    rewards = []

    for i in range(len(completions)):
        # expect conversational format
        assert completions[i][-1]["role"] == "assistant", ValueError(
            "Unexpected message: {}".format(completions)
        )

        content = completions[i][-1]["content"].strip()

        # Find the last </think> (case-insensitive)
        last_end = None
        for m in re.finditer(r"</think\s*>", content, flags=re.IGNORECASE):
            last_end = m.end()

        if last_end is not None:
            tail = content[last_end:]
            ok = bool(tail.strip())
        else:
            # No think tags: any non-empty content is valid
            ok = bool(content)

        rewards.append(1 if ok else 0)

    return rewards


def similarity_reward(prompts, completions, query, **kwargs):
    rewards = []

    for i in range(len(prompts)):
        # expect conversational format
        assert completions[i][-1]["role"] == "assistant", ValueError(
            "Unexpected message: {}".format(completions)
        )

        completion = completions[i][-1]["content"]
        generated_query = extract_answer(completion)

        reward = query_similarity(generated_query, query[i])

        rewards.append(reward)

    return rewards


def query_length_reward(prompts, completions, query, **kwargs):
    rewards = []

    for i in range(len(prompts)):
        # expect conversational format
        assert completions[i][-1]["role"] == "assistant", ValueError(
            "Unexpected message: {}".format(completions)
        )

        completion = completions[i][-1]["content"]
        generated_query = extract_answer(completion)

        reward = query_length(generated_query, query[i])

        rewards.append(reward)

    return rewards


def ef_reward(prompts, completions, answer, **kwargs):
    rewards = []

    for i in range(len(prompts)):
        # expect conversational format
        assert completions[i][-1]["role"] == "assistant", ValueError(
            "Unexpected message: {}".format(completions)
        )

        completion = completions[i][-1]["content"]
        generated_query = extract_answer(completion)

        ef = execution_feedback(generated_query, answer[i])

        rewards.append(ef)

    return rewards


def length_reward(completion_ids, **kwargs):
    rewards = []

    # max_completion_length
    a = 1024
    # encouraged length for thinking
    n = 768

    for ids in completion_ids:
        x = len(ids)
        # linear reward from 1 -> 0
        # in the interval n -> a tokens
        r = -(1 / (a - n)) * (x - n) + 1

        r = max(min(r, 1), 0)
        rewards.append(r)

    return rewards


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


log("Loading dataset")
torch_ds_train = DblpDataset(
    DATASET_PATH,
    subset="train",
    n=N_SAMPLES,
    system_prompt=SYSTEM_PROMPT,
    resource_details=True,
    query_types=query_types,
)

ds_train = HFDataset.from_list(list(torch_ds_train))

log(f"Loading tokenizer from {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

log("Loading config")
config = AutoConfig.from_pretrained(BASE_MODEL)

log("Loading model")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    config=config,
    # device_map="auto",
    trust_remote_code=True,
)


training_args = GRPOConfig(
    output_dir=MODEL_OUTPUT,
    logging_steps=5,
    log_completions=False,
    logging_dir=LOGGING_DIR,
    max_completion_length=1024,
    max_prompt_length=668,
    remove_unused_columns=False,
    num_generations=4,
    gradient_accumulation_steps=16,
    per_device_train_batch_size=4,
    report_to=["tensorboard"],
    reward_weights=WEIGHTS,
    num_train_epochs=1,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    learning_rate=0.000001,
    save_steps=20,
    min_p=0,
    beta=0.04,
    epsilon=0.2,
    bf16=True,
    # chat_template_kwargs={"enable_thinking": True},
    # use_vllm=True,
    # vllm_mode="colocate",
    # vllm_gpu_memory_utilization=1.0,
)

trainer = GRPOTrainer(
    # tokenizer=tokenizer, # is automatically loaded for processing_class -> from_pretrained(model)
    model=model,
    reward_funcs=[
        ef_reward,
        # similarity_reward,
        structure_reward,
        format_reward,
        length_reward,
        # query_length_reward
    ],
    args=training_args,
    train_dataset=ds_train,
)

log("Starting training...")
trainer.train()

query_cache.save_to_disk()

# model.save_pretrained("qwen3_4b_sparql_lora")
# tokenizer.save_pretrained("qwen3_4b_sparql_lora")
log("Done.")
