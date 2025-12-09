# GRPO for KGQA over DBLP

This admittedly messy repository contains the code for training an LLM on Knowledge Graph Question Answering using [Group-Relative Policy Optimization](https://arxiv.org/abs/2402.03300) and the [DBLP_QuAD](https://huggingface.co/datasets/awalesushil/DBLP-QuAD) dataset.


```py
├── data                            # DBLP-QuAD Dataset from HF awalesushil/DBLP-QuAD
│   ├── test                        # test split + references + metadata
│   ├── train                       # train split + references + metadata
│   ├── valid                       # valid split + references + metadata
│   ├── README.md
│   └── schema.rdf                  # schema for relation expansion during preprocessing
├── experiments                     # folder to store individual experiments
├── jobs                            # optional job outputs
├── models                          # place to store the base models
├── slurm                           # slurm scripts for training and inference
├── src                                 
│   ├── DBLPDataset.py              # torch Dataset for DBLP, preprocessing + collator
│   ├── evaluate.py                 # evaluation of inference results
│   ├── inference.py                # perform batched inference for a given model
│   ├── __init__.py
│   ├── model_client.py             # base class for a model client + decorator
│   ├── openai_endpoint.py          # openai implementation for model client
│   ├── QueryCache.py               # Query Cache impl to reduce SPARQL endpoint load
│   ├── qwen_grpo.py                # main GRPO training logic + reward functions
│   ├── qwen_local.py               # local model implementation for model client
│   ├── qwen_sft.py                 # DoRA /LoRA finetuning script
│   ├── regenerate_rdf_details.py   # regenerate dblp metadata for relation/entity expansion
│   ├── regenerate_references.py    # regenerate all answer references for the DS
│   ├── sparql.py                   # sparql endpoint wrapper
│   └── utils.py                    # some common functions
├── error_analysis.ipynb            # sample 100 incorrect outputs for manual error analysis
├── new_exp.sh                      # create a new experiment folder structure
├── README.md                       # (you are here)
└── requirements.txt                # dependencies
```

