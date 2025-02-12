from datasets import load_from_disk
import logging

# 设置日志配置（可选）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 加载保存的数据集
dataset_path = "/Users/lynn/generate-cot/fingpt_with_cot"  # 提供保存数据集的目录路径
dataset = load_from_disk(dataset_path)

# 确认数据集已正确加载
logging.info(f"Dataset loaded from {dataset_path}")

# 打印数据集中前10条记录
for i in range(min(10, len(dataset['train']))):  # 使用min确保不会超出数据集长度
    record = dataset['train'][i]
    logging.info(f"Record {i + 1}:")
    logging.info(f"Prompt: {record.get('prompt', 'N/A')}")
    logging.info(f"Answer: {record.get('answer', 'N/A')}")
    logging.info(f"COT: {record.get('cot', 'N/A')}")
    logging.info(f"label: {record.get('label', 'N/A')}")
    logging.info(f"period: {record.get('period', 'N/A')}")
    logging.info(f"symbol: {record.get('symbol', 'N/A')}")

    logging.info("-" * 40)  # 分隔线，便于阅读