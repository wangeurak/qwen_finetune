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
import inspect
# ----------------------
# 配置参数（按需修改）
# ----------------------
DATASET_PATH = r"D:\Qwen\huatuo_encyclopedia_qa"  # 数据集路径
MODEL_NAME = r"D:\Qwen\Qwen2.5-0.5B-Instruct"        # 模型名称（会自动从ModelScope下载）
SAVE_DIR = r"D:\Qwen\qwen_huatuo_lora"                  # 微调后模型保存路径

# 量化配置（显存<8GB时启用）
use_4bit = True
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# ----------------------
# 数据预处理
# ----------------------
def format_instruction(example):
    """将数据转换为Qwen指令格式"""
    return {
        "text": f"<|im_start|>system\n你是一个医疗百科助手<|im_end|>\n"
                f"<|im_start|>user\n{example['questions']}<|im_end|>\n"
                f"<|im_start|>assistant\n{example['answers']}<|im_end|>"
    }

# 加载本地数据集
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

# ----------------------
# 加载模型和分词器
# ----------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    model_max_length=1024,  # 设置最大长度
    padding_side="left"  # 生成时需左对齐
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config if use_4bit else None,
    trust_remote_code=True
)

# ----------------------
# LoRA配置
# ----------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    inference_mode=False,  # 训练模式
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 输出可训练参数量（应≈0.5%）

# ----------------------
# 训练配置
# ----------------------
training_args = TrainingArguments(
    output_dir=SAVE_DIR,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=50,
    eval_steps=200,                # 保留 eval_steps
    save_steps=500,
    save_total_limit=2,            # 限制保存的模型数量
    fp16=not use_4bit,
    report_to="wandb",
    run_name="qwen_huatuo_finetune",  # 自定义运行名称
    logging_dir="./logs",              # 日志目录
)
# print(inspect.signature(SFTTrainer.__init__))
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,        # 替换为 processing_class
)

# ----------------------
# 开始训练
# ----------------------
trainer.train()

# ----------------------
# 保存模型
# ----------------------
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

# ----------------------
# 推理测试
# ----------------------
def generate_response(query):
    prompt = (
        f"<|im_start|>system\n你是一个医疗百科助手<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant\n")[-1]

# 测试样例
print(generate_response("如何预防高血压？"))