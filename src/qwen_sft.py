import os
import time
import json

import torch
import dotenv
from datasets import Dataset as HFDataset
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer, apply_chat_template
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from DBLPDataset import DblpDataset

print("CUDA available:", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
torch.cuda.empty_cache()
print("Emptied cache")

dotenv.load_dotenv()
DATASET_PATH = os.getenv("DATASET_PATH")
BASE_MODEL = os.getenv("BASE_MODEL")
EXPERIMENT = os.getenv("EXPERIMENT")

if None in (DATASET_PATH, BASE_MODEL, EXPERIMENT):
    raise ValueError("Settings not found. Check .env file")

DATASET_PATH = os.path.expanduser(DATASET_PATH)
BASE_MODEL = os.path.expanduser(BASE_MODEL)
MODEL_OUTPUT = f"{EXPERIMENT.rstrip('/')}/checkpoints"
LOGGING_DIR = f"{EXPERIMENT.rstrip('/')}/logs"

N_SAMPLES = None  # all
RESOURCE_DETAILS = True

SYSTEM_PROMPT = """You are a SPARQL expert.
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


def log(msg):
    print(f"{time.strftime('%Y%m%d-%H%M%S')} - {msg}")


def format_prompt(sample: dict) -> str:
    entities = sample["entities_detailed"] if RESOURCE_DETAILS else sample["entities"]
    relations = sample["relations_detailed"] if RESOURCE_DETAILS else sample["relations"]

    user_msg = (
        "Generate a SPARQL query to answer the following question.\n\n"
        f"Question: {sample['question']}\n\n"
        f"Relevant Entities:\n {', '.join(entities)}\n\n"
        f"Relevant Relations:\n {', '.join(relations)}"
    )
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": sample["query"]}  # target
    ]

    return tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=False,  # loss only on assistant part by default
        tokenize=False,
        enable_thinking=False
    )


log("Loading dataset")
torch_ds_train = DblpDataset(
    DATASET_PATH,
    subset="train",
    n=N_SAMPLES,
    system_prompt=SYSTEM_PROMPT,
    resource_details=RESOURCE_DETAILS,
    include_completion_in_prompt=True,
)
ds_train = HFDataset.from_list(list(torch_ds_train))

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    use_dora=True
)

sft_config = SFTConfig(
    output_dir=MODEL_OUTPUT,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    # max_seq_length=2048,
    # assistant_only_loss=True,
    # packing=True,
    eos_token="<|im_end|>",
    num_train_epochs=4,
    logging_dir=LOGGING_DIR,
    report_to=["tensorboard"],
    # logging_steps=20,
    dataset_text_field="text",
    assistant_only_loss=False,
    completion_only_loss=False,
    learning_rate=0.00002,
    lr_scheduler_type="linear",
    weight_decay=0,
    warmup_ratio=0,
)

log(f"Loading tokenizer from {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

log("Loading config")
config = AutoConfig.from_pretrained(BASE_MODEL)

log("Loading model")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    # config=config,
    # device_map="auto",
    trust_remote_code=True,
    # peft_config=lora_config
)

# apply formatting
# ds_train = ds_train.map(format_prompt, remove_columns=["id", "prompt", "question", "entities", "relations", "query", "answer", "query_type", "entities_detailed", "relations_detailed"])
ds_train = ds_train.map(
    lambda x: {
        "text": tokenizer.apply_chat_template(
            x["prompt"],
            add_generation_prompt=False,
            tokenize=False,
            enable_thinking=False,
        )
    },
    remove_columns=["prompt"]
)
# model_peft = get_peft_model(model, lora_config)
trainer = SFTTrainer(
    model=model,
    # tokenizer=tokenizer
    train_dataset=ds_train,
    # eval_dataset=ds_eval,
    args=sft_config,
    # formatting_func=format_prompt,
    peft_config=lora_config
)

log("Starting training...")
trainer.train()
# model.save_pretrained("qwen3_4b_sparql_lora")
# tokenizer.save_pretrained("qwen3_4b_sparql_lora")
