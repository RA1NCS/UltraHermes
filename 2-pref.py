# %%
# imports
from datasets import load_dataset, Dataset
import os, random
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DPOTrainer, DPOConfig
from peft import PeftModel, prepare_model_for_kbit_training

# %%
# load environment variables
def get_env(text_file: bool = False):
    path = "/workspace/config.txt" if text_file else "/workspace/.config"
    cfg = {}

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, value = line.split("=", 1)
            cfg[key.strip()] = value.strip()

    return cfg


envars = get_env(text_file=False)


def envar(var: str, dtype: str = "str"):
    if dtype == "int":
        return int(envars.get(var.upper()))
    elif dtype == "float":
        return float(envars.get(var.upper()))
    elif dtype == "bool":
        return envars.get(var.upper()).strip().lower() in {"1", "true", "yes", "y"}
    elif dtype == "str":
        return envars.get(var.upper())

# %%
# stage 0: safety check

## device
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

## cuda
cuda_available = torch.cuda.is_available()

print(f"device: {device} | cuda: {cuda_available}")

if cuda_available:
    gpu_count = torch.cuda.device_count()
    gpu_list = [torch.cuda.get_device_name(i) for i in range(gpu_count)]

    print(f"{gpu_count} GPUs available: {gpu_list}")


## seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(envar("SEED", "int"))
print(f"seed: {envar('SEED', 'int')}")

## allow tf32

torch.backends.cuda.matmul.allow_tf32 = True

## chat template
chat_template = """
    {% set sep = '\\n\\n' -%}
    {% if bos_token is defined %}{{ bos_token }}{% endif -%}
    {%- for m in messages -%}
    {%- if m['role'] == 'system' -%}
    System:
    {{ m['content'] | trim }}{{ sep }}
    {%- elif m['role'] == 'user' -%}
    User:
    {{ m['content'] | trim }}{{ sep }}
    {%- elif m['role'] == 'assistant' -%}
    Assistant:
    {{ m['content'] | trim }}{{ sep }}
    {%- elif m['role'] == 'tool' -%}
    Tool:
    {{ m['content'] | trim }}{{ sep }}
    {%- endif -%}
    {%- endfor -%}
    {%- if add_generation_prompt -%}
    Assistant:
    {%- endif -%}
    """


# %%
# stage 1: load pref data
def convert_pref_record(record):
    prompt = [{"role": "user", "content": str(record.get("prompt", ""))}]
    
    def first_assistant(msgs):
        if isinstance(msgs, list):
            for m in msgs:
                if (m.get("role") or "").lower() == "assistant":
                    return [{"role": "assistant", "content": m.get("content", "")}]
        return [{"role": "assistant", "content": ""}]
    
    chosen = first_assistant(record.get("chosen"))
    rejected = first_assistant(record.get("rejected"))
    
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }
    

raw_pref = load_dataset(
    envar("pref_dataset"), split=f"train_prefs[:{envar('pref_sample_size')}]"
)

pref_cleaned = [convert_pref_record(r) for r in raw_pref if convert_pref_record(r) is not None]

pref_dataset = Dataset.from_dict({
    "prompt": [record["prompt"] for record in pref_cleaned],
    "chosen": [record["chosen"] for record in pref_cleaned],
    "rejected": [record["rejected"] for record in pref_cleaned],
})

pref_dataset = pref_dataset.map(
    convert_pref_record,
    remove_columns=pref_dataset.column_names,
)

# %%
# stage 2: training configs
## BitsAndBytes (QLoRA) config
qlora_config = BitsAndBytesConfig(
    load_in_4bit=envar("load_in_4bit", "bool"),
    bnb_4bit_compute_dtype={"bfloat16": torch.bfloat16, "float16": torch.float16}[
        envar("bnb_compute_dtype")
    ],
    bnb_4bit_quant_type=envar("bnb_quant_type"),
    bnb_4bit_use_double_quant=envar("bnb_double_quant", "bool"),
)

## DPO Config
dpo_config = DPOConfig(
    output_dir=os.path.join(envar("pref_save_dir"), "pref"),
    per_device_train_batch_size=envar("per_device_train_batch_size","int"),
    gradient_accumulation_steps=envar("gradient_accumulation_steps","int"),
    learning_rate=envar("learning_rate","float") * 0.5,
    warmup_ratio=envar("warmup_ratio","float"),
    weight_decay=envar("weight_decay","float"),
    num_train_epochs=envar("num_train_epochs","int"),

    # logging & saving
    logging_steps=5,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    report_to="mlflow",
    run_name="ultrahermes-pref",

    # perf/stability
    bf16=(envar("bnb_compute_dtype") == "bfloat16"),
    gradient_checkpointing=False,
    max_grad_norm=envar("grad_clip","float"),
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    dataloader_num_workers=0,
    dataloader_pin_memory=True,
    ddp_find_unused_parameters=False,
    ddp_timeout=1800,
    precompute_ref_log_probs=True,
    torch_compile=False,
    group_by_length=False,

    # nice to have in MLflow
    include_tokens_per_second=True,

    # DPO-specific
    beta=0.1,
    max_prompt_length=512,
    max_completion_length=envar("max_seq_len","int") - 512,
    generate_during_eval=False,
)

    

# %%
# stage 3: load tokenizer, policy and reference models

## 1. load tokenizer
tokenizer = AutoTokenizer.from_pretrained(envar("sft_save_dir"))

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

tokenizer.chat_template = chat_template

## 2. load model
def load_model():
    return AutoModelForCausalLM.from_pretrained(
        envar("model_id"),
        # device_map="auto", # auto for single GPU
        quantization_config=qlora_config,
        attn_implementation=envar("attn_implementation"),
    )
    
### policy model preparation
policy_model = prepare_model_for_kbit_training(load_model())
policy_model = PeftModel.from_pretrained(policy_model, envar("sft_save_dir"), is_trainable=True)
policy_model.train()
policy_model.config.use_cache = False
policy_model.config.pad_token_id = tokenizer.pad_token_id

print("policy model loaded")

### reference model preparation
reference_model = load_model()
reference_model = PeftModel.from_pretrained(reference_model, envar("sft_save_dir"), is_trainable=False)
reference_model.eval()

print("reference model loaded")

### parameter count (sanity)
trainable = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in policy_model.parameters())

print(f"Trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({100*trainable/total:.2f}%)")


# %%
# Freeze DoRA magnitude vectors to avoid multiple grad hooks
for n, p in policy_model.named_parameters():
    if "lora_magnitude_vector" in n:
        p.requires_grad = False

# stage 4: build trainer
trainer = DPOTrainer(
    model=policy_model,
    ref_model=reference_model,
    args=dpo_config,
    train_dataset=pref_dataset,
    processing_class=tokenizer,
)

# DDP workaround for LoRA parameters
if hasattr(trainer.model, "module"):
    trainer.model.module._set_static_graph()
elif hasattr(trainer.model, "_set_static_graph"):
    trainer.model._set_static_graph()

# %%
# stage 5: MLflow setup and training
trainer.train()

# %%
# stage 6: save model
policy_model.save_pretrained(envar("pref_save_dir"))
tokenizer.save_pretrained(envar("pref_save_dir"))


