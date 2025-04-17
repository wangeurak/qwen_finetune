import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_from_disk
from transformers import BitsAndBytesConfig
import os
import random
import numpy as np

# ----------------------
# 设置随机种子以确保可复现性
# ----------------------
def set_seed(seed):
    random.seed(seed)  # 设置 Python 随机种子
    np.random.seed(seed)  # 设置 NumPy 随机种子
    torch.manual_seed(seed)  # 设置 PyTorch CPU 随机种子
    torch.cuda.manual_seed_all(seed)  # 设置 PyTorch GPU 随机种子
    # 确保 CuDNN 使用确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# 设置固定的随机种子
SEED = 42
set_seed(SEED)

# ----------------------
# 配置参数（与训练时一致）
# ----------------------
MODEL_NAME = r"D:\Qwen\Qwen2.5-0.5B-Instruct"  # 原始模型路径
SAVE_DIR = r"D:\Qwen\qwen_huatuo_lora_test1"         # 微调模型保存路径
use_4bit = True  # 是否使用 4-bit 量化

# 量化配置（与训练时一致）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# ----------------------
# 加载分词器
# ----------------------
tokenizer = AutoTokenizer.from_pretrained(
    SAVE_DIR,  # 从保存路径加载分词器
    trust_remote_code=True,
    model_max_length=1024,
    padding_side="right"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
# ----------------------
# 加载基础模型
# ----------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,  # 加载原始 Qwen2.5-0.5B-Instruct 模型
    device_map="auto",
    quantization_config=bnb_config if use_4bit else None,
    trust_remote_code=True
)

# ----------------------
# 加载 LoRA 适配器
# ----------------------
# model = PeftModel.from_pretrained(
#     base_model,
#     SAVE_DIR,  # 从保存路径加载 LoRA 适配器
#     is_trainable=False,  # 推理模式
#     device_map="auto"
# )

# 确保模型处于评估模式
model.eval()

# ----------------------
# 推理函数（与原始代码一致）
# ----------------------
def generate_response(query):
    prompt = (
        f"<|im_start|>system\n你是一个医疗百科助手<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],  # 显式传递 attention_mask
        max_new_tokens=1000,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant\n")[-1]

# ----------------------
# 测试推理
# ----------------------
test_queries = [
    "如何预防高血压？",
    "曲匹地尔片的用法用量是什么？",
    "三期梅毒多久能治愈？"
]

for query in test_queries:
    print(f"\n问题: {query}")
    response = generate_response(query)
    print(f"回答: {response}")
    print("-" * 50)