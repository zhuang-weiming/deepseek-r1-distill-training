from datasets import load_from_disk, DatasetDict, Dataset
from google import genai
import logging

# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_client(api_key):
    return genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

def generate_cot_with_gemini(prompt, client):
    model = "gemini-2.0-flash-thinking-exp"
    
    try:
        response = client.models.generate_content(model=model, contents=prompt)
        api_response = response.text

        # 假设API返回的内容是通过'cot<think>'分割的answer和cot
        parts = api_response.split('cot<think>', 1)
        if len(parts) == 2:
            answer, cot = parts
        else:
            logging.warning("API response does not contain expected delimiter, defaulting to full response for both fields.")
            answer, cot = api_response, api_response
        
        logging.info(f"API Response (Answer): {answer}")
        logging.info(f"API Response (COT): {cot}")
        
        return answer, cot
    except Exception as e:
        logging.error(f"Error during API call: {e}")
        return '', ''

def add_cot_to_record(record, client):
    prompt = record['prompt'] + "并且把你的思路链内容翻译成中文，给出你是怎么分析的，只需要中文版本的放在回答后面，格式为 cot<think>: ... "
    answer, cot = generate_cot_with_gemini(prompt, client)
    record['answer'] = answer
    record['cot'] = cot
    return record

def process_missing_cot_records(dataset, client, recursion_depth=0, max_recursion_depth=10):
    missing_cot_records = [record for record in dataset['train'] if not record.get('cot') or record.get('cot').strip() in ['', 'N/A']]
    
    if not missing_cot_records:
        logging.info("No more records with missing COT values.")
        return dataset
    
    if recursion_depth >= max_recursion_depth:
        logging.warning(f"Reached maximum recursion depth ({max_recursion_depth}). Stopping further processing.")
        return dataset

    logging.info(f"Processing {len(missing_cot_records)} records with missing COT values at recursion depth {recursion_depth}.")
    
    updated_train_data = []
    for i, record in enumerate(dataset['train']):
        if not record.get('cot') or record.get('cot').strip() in ['', 'N/A']:
            logging.info(f"Processing missing COT record {i + 1}/{len(dataset['train'])}")
            updated_record = add_cot_to_record(dict(record), client)
            updated_train_data.append(updated_record)
        else:
            updated_train_data.append(record)

    updated_dataset = Dataset.from_list(updated_train_data)
    dataset_dict = DatasetDict({'train': updated_dataset})

    # 递归调用自身以继续处理可能仍然存在的缺失值
    return process_missing_cot_records(dataset_dict, client, recursion_depth + 1, max_recursion_depth)

# 加载已保存的数据集
dataset_path = "/Users/lynn/generate-cot/fingpt_with_cot_updated"  # 提供保存数据集的目录路径
dataset = load_from_disk(dataset_path)

# 初始化客户端
api_keys = ['AIzaSyB8INnHRD7KXfPdlY37X3yGBYAwhYxN1hY', 'AIzaSyAQOd9XKz5J2gEeoOkIkHqAWNy3n451Gto']
client = create_client(api_keys[0])  # 使用第一个API密钥

# 开始递归处理
final_dataset = process_missing_cot_records(dataset, client)

# 查看前几条记录以确认新字段已正确添加
logging.info(f"First record after updating missing COT fields: {final_dataset['train'][0]}")
logging.info(f"count: {len(final_dataset['train'])}")

# 保存修改后的数据集到本地文件
final_dataset.save_to_disk("fingpt_combined")
logging.info("Updated dataset saved successfully.")