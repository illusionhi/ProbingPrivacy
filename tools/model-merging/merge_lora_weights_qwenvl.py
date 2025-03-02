import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 设置模型路径
base_model_dir = ""  # 替换为原始Qwen-VL-Chat模型的路径
lora_model_dir = "./checkpoints/qwenvl"  # 替换为微调后的LoRA模型路径
merged_model_dir = ""  # 替换为你希望保存合并后模型的路径

# 指定使用的GPU设备（例如，第一个GPU）
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 加载原始模型和分词器
try:
    print("Loading base model and tokenizer...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": device},  # 指定使用的设备
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
except Exception as e:
    print(f"Error loading base model or tokenizer: {e}")
    raise

# 加载LoRA模型
try:
    print("Loading LoRA model...")
    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_model_dir,
        device_map={"": device},  # 指定使用的设备
        torch_dtype=torch.float16,
    )
except Exception as e:
    print(f"Error loading LoRA model: {e}")
    raise

# 合并LoRA权重到原始模型
try:
    print("Merging LoRA weights into base model...")
    merged_model = lora_model.merge_and_unload()
except Exception as e:
    print(f"Error merging LoRA model: {e}")
    raise

# 保存合并后的模型
try:
    print(f"Saving merged model to {merged_model_dir}...")
    merged_model.save_pretrained(merged_model_dir, max_shard_size="1GB")
    tokenizer.save_pretrained(merged_model_dir)
    print("Model merging complete.")
except Exception as e:
    print(f"Error saving merged model: {e}")
    raise
