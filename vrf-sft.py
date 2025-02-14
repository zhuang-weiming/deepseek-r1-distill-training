import torch
from unsloth import FastLanguageModel

model_name = "unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit"
max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype = None,
    load_in_4bit=True,
)

prompt_style = """
以下是描述任务的说明，并附带提供更多上下文的输入。请写出适当完成请求的回复。
回答之前，请仔细思考问题，并构建循序渐进的思路，以确保做出合乎逻辑且准确的预测。

### 任务说明：
您是一位资深金融分析师，精通股票交易、经济分析和市场趋势预测。请根据下面的问题提示以及额外上下文，做出预测性分析。预测回答需包括两部分：
1. **预测答案**：针对问题的详细解答和分析结论。
2. **预期波动范围**：预测答案中涉及的股票价格波动幅度。

### 问题：
{}

### 回复：
{}
"""

question = "2023年1月贵州茅台的股票走势"

# Do model patching and add fast LoRA weights
# model = FastLanguageModel.get_peft_model(
#     model,
#     r = 16,
#     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
#                       "gate_proj", "up_proj", "down_proj",],
#     lora_alpha = 16,
#     lora_dropout = 0, # Supports any, but = 0 is optimized
#     bias = "none",    # Supports any, but = "none" is optimized
#     # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
#     use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
#     random_state = 3407,
#     max_seq_length = max_seq_length,
#     use_rslora = False,  # We support rank stabilized LoRA
#     loftq_config = None, # And LoftQ
# )

# 加载微调后的 adapter 权重
# adapter_state_dict = torch.load("./outputs/checkpoint-60/training_args.bin", map_location="cuda")
# model.load_state_dict(adapter_state_dict)

# 2. 加载微调得到的适配器权重（outputs/checkpoint-60中仅保存了LoRA参数）
# unsloth 的微调流程只保存了 adapter 参数，您需要将其加载到大模型中。
# 假设在 checkpoint 文件夹中保存了 adapter 的 state_dict（如 adapter.bin）
adapter_path = "outputs/checkpoint-60/adapter.bin"
adapter_state = torch.load(adapter_path, map_location="cpu")
# 加载时使用 strict=False，因为只更新了部分参数
model.load_state_dict(adapter_state, strict=False)

FastLanguageModel.for_inference(model)
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

# 生成文本
outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1200,  # 设置生成的最大 token 数量
    use_cache=True
)

# 解码生成的文本
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)