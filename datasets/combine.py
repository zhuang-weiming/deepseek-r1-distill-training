from datasets import load_from_disk, DatasetDict, concatenate_datasets
import logging

# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 加载已保存的数据集
dataset_path_dow30 = "/Users/lynn/generate-cot/fingpt_with_cot_train_dow30_v3_1"  # 提供保存数据集的目录路径
dataset_dow30 = load_from_disk(dataset_path_dow30)

dataset_path_sz50 = "/Users/lynn/generate-cot/fingpt_with_cot_train_sz50_v3_1"  # 提供保存更新后数据集的目录路径
dataset_sz50 = load_from_disk(dataset_path_sz50)

# 合并两个数据集
combined_dataset = concatenate_datasets([dataset_sz50['train'], dataset_dow30['train']])
logging.info(f"Combined dataset contains {len(combined_dataset)} records.")

# 创建一个新的 DatasetDict 对象
final_dataset_dict = DatasetDict({'train': combined_dataset})

# 查看前几条记录以确认合并结果
logging.info(f"First record after merging datasets: {final_dataset_dict['train'][0]}")
logging.info(f"count : {len(final_dataset_dict['train'])}")

# 保存合并后的数据集到本地文件
final_dataset_dict.save_to_disk("fingpt_cot_combined_v1")
logging.info("Combined dataset saved successfully.")