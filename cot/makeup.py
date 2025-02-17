from datasets import load_from_disk, DatasetDict, Dataset
from google import genai
import logging
from datetime import datetime


# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def construct_detailed_prompt(record):
    # prompt version 1
    # detailed_prompt = origin_prompt + "并且把你的思路链内容翻译成中文，给出你是怎么分析的，只需要中文版本的放在回答后面，格式为 cot<think>: ..."

    # prompt version 2
    detailed_prompt = f"""
Prompt：

{record['prompt']}

Answer：

{record['answer']}

Period：

{record['period']}

Label：

{record['label']}

Symbol：

{ record['symbol']}

请帮忙升级出针对本条数据的CoT （思维链条）的内容，并升级answer的内容。你的回答语言应为中文，预测涨跌幅应该等于label。你的回答格式应该如下：

CoT：
[信息整合与解读]: …..

[积极发展因素分析]: 
1 … (积极发展因素分析1)
2 … (积极发展因素分析2)

[潜在担忧因素分析]
1 … (潜在担忧因素分析1)
2 … (潜在担忧因素分析2)

[股价预测与分析总结]
    * **综合权衡：**  ...
    * **预测涨跌幅：**  …
    * **总结分析：**  ...

总而言之，整个思维链条遵循了从信息收集、分类解读，到因素分析、权衡判断，最终形成预测和总结的完整过程，力求基于客观信息和合理推演，得出相对严谨的分析结论。

升级Answer：

 [积极发展]： 
1. … 

[潜在担忧]： 
1. …

[预测和分析]： 
预测涨跌幅：… 
总结分析：... 

"""
    return detailed_prompt

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
        logging.info(f"API Response: {api_response}")

        # prompt version 1
        # 假设API返回的内容是通过'cot<think>'分割的answer和cot
        # parts = api_response.split('cot<think>', 1)
        # parts = api_response.split('升级Answer', 1)
        # if len(parts) == 2:
        #     cot, answer = parts
        # else:
        #     logging.warning("API response does not contain expected delimiter, defaulting to full response for both fields.")
        #     cot, answer = api_response, api_response

        # prompt version 2
        # 提取answer部分，从'升级Answer'到'period'之前
        answer_start = api_response.find("升级Answer：")
        period_start = api_response.find("Period：")
        if answer_start != -1:
            if period_start != -1:
                answer = api_response[answer_start:period_start].strip()
            else:
                # 如果找到了'升级Answer：'但未找到'Period：'
                answer = api_response[answer_start:].strip()
        else:
            logging.warning("未能找到升级Answer标记，使用完整响应作为answer")
            answer = api_response
            
        # 提取cot部分，从'cot'开始到'answer'之前
        cot_start = api_response.find("CoT：")
        cot_end = api_response.find("升级Answer：")
        if cot_start != -1 and cot_end != -1:
            cot = api_response[cot_start:cot_end].strip()
        else:
            logging.warning("未能找到CoT或升级Answer标记，使用完整响应作为cot")
            cot = api_response
        
        logging.info(f"API Response (Answer): {answer}")
        logging.info(f"API Response (COT): {cot}")
        
        return cot, answer
    except Exception as e:
        logging.error(f"Error during API call: {e}")
        return '', ''

def add_cot_to_record(record, client):
    # 提取 [公司介绍]: 后面的内容
    # company_info_start = record['prompt'].find("[公司介绍]:")
    # if company_info_start == -1:
    #     logging.error("未找到 '[公司介绍]:' 标记")
    #     return record
    
    # company_info_end = record['prompt'].find("基于在", company_info_start)
    # if company_info_end == -1:
    #     logging.error("未找到 '接下来请预测' 标记")
    #     return record

    # company_info = record['prompt'][company_info_start:company_info_end].strip()

    prompt = construct_detailed_prompt(record)
    # logging.info(f"prompt: {prompt}") 

    cot, answer = generate_cot_with_gemini(prompt, client)
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
dataset_path = "/Users/lynn/generate-cot/fingpt_with_cot_train_sz50_v3"  # 提供保存数据集的目录路径
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
final_dataset.save_to_disk("fingpt_with_cot_train_sz50_v3_1")
logging.info("Updated dataset saved successfully.")