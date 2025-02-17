from datasets import load_from_disk
import logging

# 设置日志配置（可选）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 加载保存的数据集
dataset_path = "/Users/lynn/generate-cot/fingpt_cot_combined_v1"  # 提供保存数据集的目录路径
dataset = load_from_disk(dataset_path)

# 确认数据集已正确加载
logging.info(f"Dataset loaded from {dataset_path}")

test_subset = dataset['train']
logging.info(f"The train subset contains {len(test_subset)} records.")

# # 打印数据集中前10条记录
# for i in range(min(5, len(dataset['test']))):  # 使用min确保不会超出数据集长度
#     record = dataset['test'][i]
#     logging.info(f"Record {i + 1}:")
#     logging.info(f"Prompt: {record.get('prompt', 'N/A')}")
#     logging.info(f"Answer: {record.get('answer', 'N/A')}")
#     logging.info(f"COT: {record.get('cot', 'N/A')}")
#     logging.info(f"label: {record.get('label', 'N/A')}")
#     logging.info(f"period: {record.get('period', 'N/A')}")
#     logging.info(f"symbol: {record.get('symbol', 'N/A')}")

#     logging.info("-" * 40)  # 分隔线，便于阅读


missing_cot_records = []

# 遍历数据集，寻找 cot 字段为空的记录
for i, record in enumerate(test_subset):
    if not record.get('cot') or record.get('cot').strip() in ['', 'N/A']:
        missing_cot_records.append(record)

# 打印出没有 cot 数值的记录
# for i, record in enumerate(missing_cot_records):
#     logging.info(f"Record {i + 1}:")
#     logging.info(f"Prompt: {record.get('prompt', 'N/A')}")
#     logging.info(f"Answer: {record.get('answer', 'N/A')}")
#     logging.info(f"COT: {record.get('cot', 'N/A')}")
#     logging.info(f"Period: {record.get('period', 'N/A')}")
#     logging.info(f"Label: {record.get('label', 'N/A')}")
#     logging.info(f"Symbol: {record.get('symbol', 'N/A')}")
#     logging.info("-" * 40)  # 分隔线，便于阅读

# 计算并打印没有 cot 数值的记录数量
logging.info(f"Total number of records without COT value: {len(missing_cot_records)}")