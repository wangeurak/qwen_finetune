import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import pandas as pd
import re

# 配置参数
DATASET_PATH = r"D:\Qwen\huatuo_encyclopedia_qa"
MODEL_NAME = r"D:\Qwen\Qwen2.5-0.5B-Instruct"
SAVE_DIR = r"D:\Qwen\qwen_huatuo_lora"
use_4bit = True
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 数据预处理
def format_instruction(example):
    import re
    question = example['questions']
    if isinstance(question, list) and len(question) > 0:
        flat_questions = [q for sublist in question for q in (sublist if isinstance(sublist, list) else [sublist])]
        question = flat_questions[0] if flat_questions else ""
    question = re.sub(r'\[.*?\]', '', question)
    question = re.sub(r'\s+', ' ', question).strip()

    answer = example['answers']
    if isinstance(answer, list) and len(answer) > 0:
        answer = answer[0]
    answer = re.sub(r'\n+', '\n', answer)
    answer = re.sub(r'\s+', ' ', answer).strip()

    return {
        "text": f"<|im_start|>system\n你是一个医疗百科助手<|im_end|>\n"
                f"<|im_start|>user\n{question}<|im_end|>\n"
                f"<|im_start|>assistant\n{answer}<|im_end|>\n"
    }

dataset = load_dataset(
    "json",
    data_files={
        "train": os.path.join(DATASET_PATH, "train_datasets.jsonl"),
        "validation": os.path.join(DATASET_PATH, "validation_datasets.jsonl")
    }
)
dataset = dataset.map(format_instruction, remove_columns=dataset["train"].column_names)
dataset["train"] = dataset["train"].select(range(2000))
dataset["validation"] = dataset["validation"].select(range(200))
print("Train dataset samples:")
for i, text in enumerate(dataset["train"].select(range(5))["text"]):
    print(f"样本 {i}: {text}\n{'-'*50}")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    model_max_length=1024,
    padding_side="right"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token or "<|PAD|>"
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    print('break')

print(f"原始 pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")

lengths = [len(tokenizer.encode(sample["text"])) for sample in dataset["train"]]
print(f"Train dataset stats: Min length={min(lengths)}, Max length={max(lengths)}, Mean length={sum(lengths)/len(lengths)}")

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config if use_4bit else None,
    trust_remote_code=True,
    attn_implementation="eager",
    sliding_window=None  # 禁用滑动窗口
)

# LoRA 配置
lora_config = LoraConfig(
    r=4,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    inference_mode=False,
)
model = get_peft_model(model, lora_config)

# 训练配置
training_args = TrainingArguments(
    output_dir=SAVE_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
    learning_rate=5e-5,
    num_train_epochs=1,
    logging_steps=10,
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    fp16=not use_4bit,
    report_to="wandb",
    run_name="qwen_huatuo_finetune",
    logging_dir="./logs",
)

# 初始化 SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
)

# 开始训练
trainer.train()

# 保存模型
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)