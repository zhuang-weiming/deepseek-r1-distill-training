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
[INST]你是一名资深的股票市场分析师。请基于以下【公司介绍】、【股票价格变动】、【公司新闻】和【金融基本面信息】，分析公司的【积极发展】和【潜在担忧】，并【预测和分析】该公司未来一周的股价涨跌幅。

**请严格按照以下格式输出：**

[积极发展]：
1. ...
2. ...
3. ...

[潜在担忧]：
1. ...
2. ...
3. ...

[预测和分析]：
预测涨跌幅：...
总结分析：...

**以下是分析所需的信息：**

**[公司介绍]**：
<公司介绍>

**[股票价格变动]** (过去一周)：
<股票价格变动>

**[公司新闻]** (过去一周)：
<新闻标题1>: <新闻内容1>
<新闻标题2>: <新闻内容2>
...

**[金融基本面信息]** (报告期: <报告期>)：
<金融基本面信息>

请开始你的分析。
{}[/INST]
"""

question = "请分析2023年1月贵州茅台的股票走势"

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
adapter_path = "./outputs/checkpoint-60"
model.load_adapter(adapter_path)

FastLanguageModel.for_inference(model)
inputs = tokenizer([prompt_style.format(question)], return_tensors="pt").to("cuda")

# 生成文本
outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=50000,  # 设置生成的最大 token 数量
    use_cache=True
)

# 解码生成的文本
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)