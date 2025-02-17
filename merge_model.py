from peft import PeftModel
from unsloth import FastLanguageModel 

# 设置模型的最大序列长度
max_seq_length = 2048 # 模型支持 RoPE Scaling，可根据需要设置任意长度

# 加载预训练模型和分词器
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit", # 指定模型名称
    max_seq_length = max_seq_length, # 设置最大序列长度
    dtype = None, # 自动检测并设置数据类型
    load_in_4bit = True, # 以 4-bit 精度加载模型
)

adapter_path = "./outputs/3-fingpt-data-1813-update-prompts"  # 你的 adapter 文件所在目录
# 注意 adapter_name 通常默认是 "default"，如果你使用其他名称，请相应修改
model_with_adapter = PeftModel.from_pretrained(model, adapter_path, adapter_name="default")

merged_model = model_with_adapter.merge_and_unload()
merged_model.save_pretrained_gguf("gguf_model_dir", tokenizer, quantization_method="q4_k_m")
