import time

from unsloth import FastLanguageModel 
from unsloth import is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 记录程序开始时间
start_time = time.time()

# 定义训练时使用的提示模板
train_prompt_style = """
<｜begin▁of▁sentence｜>
<｜User｜>
**记录时间：**{}
**股票代码：**{}
{}
<｜Assistant｜>
<think>
{}
</think>
{}
**预测判断：**
{}
<｜end▁of▁sentence｜>
"""

# 设置模型的最大序列长度
max_seq_length = 2048 # 模型支持 RoPE Scaling，可根据需要设置任意长度

# 加载预训练模型和分词器
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit", # 指定模型名称
    max_seq_length = max_seq_length, # 设置最大序列长度
    dtype = None, # 自动检测并设置数据类型
    load_in_4bit = True, # 以 4-bit 精度加载模型
)

# 定义函数，将数据集中的各列格式化为模型输入所需的文本格式
def formatting_prompts_func(examples):
    prompts = examples["prompt"]
    answers = examples["answer"]
    cots = examples["cot"]
    periods = examples["period"]
    labels = examples["label"]
    symbols = examples["symbol"]
    texts = []
    # 遍历每一行数据，按照模板格式化文本
    for prompt, answer, cot, period, label, symbol in zip(prompts, answers, cots, periods, labels, symbols):
        text = train_prompt_style.format(period, symbol, prompt, cot, answer, label)
        texts.append(text)
    return {
        "text": texts
    }

# 加载保存的数据集
dataset_path = "LYNN2024/fingpt_with_cot_combined_v3"  # 提供保存数据集的目录路径
dataset = load_dataset(dataset_path, split = "train")

# 将数据集中的每一行应用格式化函数，生成模型输入所需的文本
dataset = dataset.map(formatting_prompts_func, batched=True)

# 检查生成结果，debug用
# for i in range(min(10, len(dataset))):  # 使用min确保不会超出数据集长度
#     record = dataset[i]
#     print(f"symbol: {record.get('text', 'N/A')}")

# 对模型进行 LoRA（低秩适配）配置，以进行高效的参数微调
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # LoRA 的秩（rank）参数
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",], # 需要应用 LoRA 的目标模块
    lora_alpha = 16, # LoRA 的 alpha 参数
    lora_dropout = 0, # LoRA 的 dropout 概率，0 表示不使用 dropout
    bias = "none",    # 不对偏置项进行微调
    use_gradient_checkpointing = "unsloth", # 使用 Unsloth 提供的梯度检查点，节省显存
    random_state = 3407, # 随机种子，确保结果可复现
    use_rslora = False,  # 是否使用 Rank Stabilized LoRA
    loftq_config = None, # 是否使用 LoFTQ 量化配置
)

# 配置训练参数并初始化 SFTTrainer
trainer = SFTTrainer(
    model = model, # 指定要训练的模型
    train_dataset = dataset, # 指定训练数据集
    dataset_text_field = "text", # 数据集中包含文本的字段名称
    max_seq_length = max_seq_length, # 设置最大序列长度
    tokenizer = tokenizer, # 指定分词器
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2, # 每个设备上的训练批次大小
        gradient_accumulation_steps = 4, # 梯度累积步数，相当于增大有效批次大小
        warmup_steps = 5, # 学习率预热步数
        max_steps = 60, # 最大训练步数
        learning_rate = 2e-5, # 学习率
        fp16 = not is_bfloat16_supported(), # 如果不支持 bfloat16，则使用 fp16
        bf16 = is_bfloat16_supported(), # 如果支持 bfloat16，则使用 bf16
        logging_steps = 1, # 日志记录的步数间隔
        output_dir = "outputs", # 模型和日志的输出目录
        optim = "adamw_8bit", # 优化器类型，使用 8-bit AdamW 优化器
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407, # 随机种子，确保结果可复现
        report_to="wandb", # 使用 Weights & Biases 进行实验跟踪
    ),
)
# 开始训练模型
trainer.train()

# 记录程序结束时间
end_time = time.time()

# 计算程序运行时间
execution_time = end_time - start_time

# 打印程序运行时间
print(f"程序运行时间：{execution_time} 秒")