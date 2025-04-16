import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

DATASET_PATH = r"D:\Qwen\huatuo_encyclopedia_qa"  # 数据集路径
MODEL_NAME = r"D:\Qwen\Qwen2.5-0.5B-Instruct"        # 模型名称（会自动从ModelScope下载）
SAVE_DIR = r"D:\Qwen\dataset"                  # 微调后模型保存路径

def format_instruction(example):
    import re
    # 处理 questions 字段（嵌套列表）
    question = example['questions']
    if isinstance(question, list) and len(question) > 0:
        # 提取第一个子列表中的第一个问题
        question = question[0][0] if isinstance(question[0], list) and len(question[0]) > 0 else ""
    question = re.sub(r'\[.*?\]', '', question).strip()  # 去除标签并清理

    # 处理 answers 字段（列表）
    answer = example['answers']
    if isinstance(answer, list) and len(answer) > 0:
        answer = answer[0]  # 提取第一个答案
    answer = re.sub(r'\n+', '\n', answer).strip()  # 清理多余换行符

    return {
        "text": f"<|im_start|>system\n你是一个医疗百科助手<|im_end|>\n"
                f"<|im_start|>user\n{question}<|im_end|>\n"
                f"<|im_start|>assistant\n{answer}<|im_end|>"
    }

dataset = load_dataset(
    "json",
    data_files={
        "train": os.path.join(DATASET_PATH, "train_datasets.jsonl"),
        "validation": os.path.join(DATASET_PATH, "validation_datasets.jsonl")
    }
)
dataset = dataset.map(format_instruction, remove_columns=dataset["train"].column_names)
dataset["train"] = dataset["train"].select(range(1000))
dataset["validation"] = dataset["validation"].select(range(200))

dataset.save_to_disk(SAVE_DIR)  # 保存预处理后的数据集
