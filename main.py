#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# imports
from datasets import load_dataset, Dataset
import os, random
import numpy as np
import torch, transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import mlflow


# In[ ]:


# load environment variables
def get_env(text_file: bool = False):
    path = "/workspace/envars.txt" if text_file else "/workspace/.env"
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


# In[ ]:


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


# In[ ]:


# stage 1: load data
def clean_sft_record(record):
    conv = record.get("conversations")

    def get_role_value(conv, role):
        return next((c["value"] for c in conv if c.get("from").lower() == role), None)

    system = get_role_value(conv, "system")
    user = get_role_value(conv, "human")
    assistant = get_role_value(conv, "gpt")

    if user and assistant:
        if system:
            text = f"System:\n{system}\n\nUser:\n{user}\n\nAssistant:\n{assistant}"
        else:
            text = f"User:\n{user}\n\nAssistant:\n{assistant}"
        return {"text": text}
    return {"text": None}


def remove_empty_text(dataset):
    return [conv["text"] for conv in dataset if conv["text"] is not None]


## raw data
raw_sft = load_dataset(
    envar("sft_dataset"), split=f"train[:{envar('sft_sample_size')}]"
)
# raw_pref = load_dataset(
#     envar("pref_dataset"), split=f"train[:{envar('pref_sample_size')}]"
# )

## normalize data
### clean data
sft_cleaned = raw_sft.map(clean_sft_record)
sft_cleaned = remove_empty_text(sft_cleaned)

### load it
sft_dataset = Dataset.from_dict({"text": sft_cleaned})

## TODO: normalization of pref data


# In[ ]:


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

## PEFT config
peft_config = LoraConfig(
    r=envar("lora_r", "int"),
    lora_alpha=envar("lora_alpha", "int"),
    lora_dropout=envar("lora_dropout", "float"),
    target_modules=[
        module.strip()
        for module in envar("target_modules").split(",")
        if module.strip()
    ],
    bias="none",
    task_type="CAUSAL_LM",
    use_dora=envar("use_dora", "bool"),
)

## TrainingArguments
training_args = TrainingArguments(
    output_dir=envar("save_dir"),
    per_device_train_batch_size=envar("per_device_train_batch_size", "int"),
    gradient_accumulation_steps=envar("gradient_accumulation_steps", "int"),
    learning_rate=envar("learning_rate", "float"),
    warmup_ratio=envar("warmup_ratio", "float"),
    weight_decay=envar("weight_decay", "float"),
    num_train_epochs=envar("num_train_epochs", "int"),
    logging_steps=25,
    save_strategy="epoch",
    bf16=(envar("bnb_compute_dtype") == "bfloat16"),
    gradient_checkpointing=False,
    max_grad_norm=envar("grad_clip", "float"),
    report_to="mlflow",
    run_name="ultrahermes",
    save_total_limit=2,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    ddp_find_unused_parameters=False,
)


# In[ ]:


# stage 3: load tokenizer and model

## 1. load tokenizer
tokenizer = AutoTokenizer.from_pretrained(envar("model_id"))

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

## 2. load model
### base preparation
base_model = AutoModelForCausalLM.from_pretrained(
    envar("model_id"),
    device_map="auto",
    quantization_config=qlora_config,
    attn_implementation=envar("attn_implementation"),
)
base_model.config.use_cache = False
base_model.config.pad_token_id = tokenizer.pad_token_id
base_model = prepare_model_for_kbit_training(base_model)

print("base model loaded")

### final model
model = get_peft_model(base_model, peft_config)

#### param info (sanity)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
percent_trainable = 100 * trainable_params / total_params

print("model loaded")
print(
    f"targets: {peft_config.target_modules} | trainable params: {trainable_params} / {total_params} ({percent_trainable:.2f}%)"
)


# In[ ]:


# stage 4: tokenize and pack data
def build_packed_dataset(token_stream, eos_id, max_token_length):
    all_ids = []
    for ids in token_stream["input_ids"]:
        if not ids:
            continue
        all_ids.extend(ids)
        if eos_id is not None:
            all_ids.append(eos_id)

    chunks = [all_ids[i:i+max_token_length] for i in range(0, len(all_ids) - max_token_length + 1, max_token_length)]
    print(f"tokens: {len(all_ids):,} | {len(chunks):,} chunks of {max_token_length} tokens each")
    return Dataset.from_dict({"input_ids": chunks})

max_token_length = envar("max_seq_len", "int")
eos_id = tokenizer.eos_token_id

## tokenize without truncation (no padding or special tokens)
def tokenize(batch):
    return tokenizer(batch["text"], add_special_tokens=False, truncation=False)

sft_stream = sft_dataset.map(tokenize, batched=True, remove_columns=["text"])

## pack data
sft_packed = build_packed_dataset(sft_stream, eos_id, max_token_length)
sft_packed = sft_packed.map(lambda ex: {
    "attention_mask": [1]*len(ex["input_ids"]),
    "labels": ex["input_ids"]
})


# In[ ]:


# stage 5: build trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=sft_packed,
    data_collator=None,
)


# In[ ]:


# stage 6: train
trainer.train()


# In[ ]:


# stage 7: save model
model.save_pretrained(envar("save_dir"))
tokenizer.save_pretrained(envar("save_dir"))

