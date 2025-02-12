import logging
from datetime import datetime
from datasets import load_dataset, DatasetDict
from google import genai

# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_client(api_key):
    return genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

def generate_cot_with_gemini(prompt, client):
    model = "gemini-2.0-flash-thinking-exp"
    
    start_time = datetime.now()
    try:
        response = client.models.generate_content(model=model, contents=prompt)
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        logging.info(f"Request completed in {elapsed_time.total_seconds()} seconds")
        api_response = response.text
        # logging.info(f"API Response: {api_response}")

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

# 加载你的数据集
dataset = load_dataset("FinGPT/fingpt-forecaster-sz50-20230201-20240101")

# 初始化客户端
api_keys = ['AIzaSyB8INnHRD7KXfPdlY37X3yGBYAwhYxN1hY', 'AIzaSyAQOd9XKz5J2gEeoOkIkHqAWNy3n451Gto']  # 替换为你的实际API密钥
clients = [create_client(key) for key in api_keys]

# 分割数据集
halfway_point = len(dataset['train']) // 2
part1 = dataset['train'].select(range(halfway_point))
part2 = dataset['train'].select(range(halfway_point, len(dataset['train'])))

# 对第一部分数据使用第一个API密钥处理
updated_train_data_part1 = []
for i, record in enumerate(part1):
    logging.info(f"Processing record {i + 1}/{len(part1)} with first API key")
    updated_record = add_cot_to_record(record, clients[0])
    updated_train_data_part1.append(updated_record)

# 对第二部分数据使用第二个API密钥处理
updated_train_data_part2 = []
for i, record in enumerate(part2):
    logging.info(f"Processing record {i + halfway_point + 1}/{len(dataset['train'])} with second API key")
    updated_record = add_cot_to_record(record, clients[1])
    updated_train_data_part2.append(updated_record)

# 合并两部分数据
updated_train_data = updated_train_data_part1 + updated_train_data_part2

# 更新数据集
dataset = DatasetDict({'train': dataset['train'].new(updated_train_data)})

# 查看前几条记录以确认新字段已正确添加
logging.info(f"First record after adding answer and cot fields: {dataset['train'][0]}")

# 保存修改后的数据集到本地文件
dataset.save_to_disk("fingpt_with_cot")
logging.info("Dataset saved successfully.")