from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 加载原始模型和分词器
base_model_path = "unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit"
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16)

# 加载微调后的 LoRA 适配器
lora_model_path = "outputs/3-fingpt-data-1813-update-prompts"
lora_model = PeftModel.from_pretrained(base_model, lora_model_path)

# 将 LoRA 适配器合并到原始模型
merged_model = lora_model.merge_and_unload()

# 保存合并后的模型
output_dir = "merged_model"
merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)