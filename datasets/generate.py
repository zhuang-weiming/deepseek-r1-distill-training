import logging
from datetime import datetime
from datasets import load_dataset, DatasetDict, Dataset, load_from_disk
from google import genai

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

# 加载你的数据集
dataset = load_dataset("FinGPT/fingpt-forecaster-sz50-20230201-20240101")

# 初始化客户端
api_keys = ['AIzaSyARu5wgFiuvKRPxXYSndwBZ9LdOoJ6DUiw', 'AIzaSyAEpdIDC1gtjPGh-LvnbsqQLKv-_CxP8Ck']  # 替换为你的实际API密钥
clients = [create_client(key) for key in api_keys]

# 分割数据集
halfway_point = len(dataset['train']) // 2

part1 = dataset['train'].select(range(halfway_point))
part2 = dataset['train'].select(range(halfway_point, len(dataset['train'])))

# 对第一部分数据使用第一个API密钥处理
updated_train_data_part1 = []
for i, record in enumerate(part1):
    logging.info(f"Processing record {i + 1}/{len(part1)} with first API key")
    updated_record = add_cot_to_record(dict(record), clients[0])  # 将record转换为dict
    updated_train_data_part1.append(updated_record)

# 对第二部分数据使用第二个API密钥处理
updated_train_data_part2 = []
for i, record in enumerate(part2):
    logging.info(f"Processing record {i + halfway_point + 1}/{len(dataset['train'])} with second API key")
    updated_record = add_cot_to_record(dict(record), clients[1])  # 将record转换为dict
    updated_train_data_part2.append(updated_record)

# 合并两部分数据
updated_train_data = updated_train_data_part1 + updated_train_data_part2

# 更新数据集
updated_dataset = Dataset.from_list(updated_train_data)

# 查看前几条记录以确认新字段已正确添加
logging.info(f"First record after adding answer and cot fields: {updated_dataset[0]}")

# 保存修改后的数据集到本地文件
dataset_dict = DatasetDict({'train': updated_dataset})
dataset_dict.save_to_disk("fingpt_with_cot_train_sz50_v2")
logging.info("Dataset saved successfully.")