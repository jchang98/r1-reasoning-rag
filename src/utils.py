from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from rich.console import Console
import asyncio
import networkx as nx
import os


console = Console()
    

def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)

def write_json(json_obj, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


def load_json(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)

def parallel_process(lst, process_element, use_threads=True, max_workers=None):
    """
    通用并行处理函数。

    参数：
    - lst: 要处理的列表。
    - process_element: 操作函数，应用到列表每个元素上。
    - use_threads: 是否使用线程池。默认 True。设置为 False 使用进程池。
    - max_workers: 并行工作的最大数量。默认 None 自动设置。

    返回值：
    - 返回处理后的新列表，顺序与输入列表一致。
    """
    executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

    with executor_class(max_workers=max_workers) as executor:
        result = list(executor.map(process_element, lst))
    
    return result

# 添加异步并行处理函数
async def async_parallel_process(items, process_func):
    """
    异步并行处理多个项目
    
    Args:
        items: 要处理的项目列表
        process_func: 处理单个项目的异步函数
        
    Returns:
        处理结果的列表
    """
    tasks = []
    for item in items:
        tasks.append(asyncio.create_task(process_func(item)))
    
    return await asyncio.gather(*tasks)

from collections import defaultdict
import re
import streamlit as st
#读取目录下所有文件名，用问题名-时间进行排列
def parse_file_names(file_names):
    problems = defaultdict(list)
    times = defaultdict(list)
    for file_name in file_names:
        parts = file_name.split('_')
        if len(parts) == 2:
            problem, time = parts[0], parts[1].split('.')[0]
            problem = problem[5:]
            problems[problem].append(time)

            times[time[:8]].append(file_name[5:].split(".")[0])
    return problems, times

def clean_logs(log_text):
    # 正则表达式模式，匹配ERROR和WARNING日志及其堆栈信息
    pattern = r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - .+? - (?:ERROR|WARNING) - .+?(?=\n^\d{4}-\d{2}-\d{2} |\Z)'
    
    # 使用re.MULTILINE匹配多行，re.DOTALL让.匹配换行符
    cleaned_text = re.sub(
        pattern,
        '',
        log_text,
        flags=re.MULTILINE | re.DOTALL
    )
    
    # 去除多余空行
    return re.sub(r'\n\s*\n', '\n', cleaned_text)

# 打印日志文件。 1.删除检索的原始信息；2.根据步骤进行划线
def show_hist_log(show_file):
    show_file = f"logs/{show_file}.log"
    try:
        with open(show_file, 'r', encoding='utf-8') as file:
            content = file.read()
            content = clean_logs(content)

            #Newly retrieved context: \n{newly_retrieved_context_lst}\nNewly useful info: {newly_useful_info
            pattern = r"'Newly retrieved context':\s*\[(.*?)\],"
            content = re.sub(pattern, "[]", content, flags=re.DOTALL)

            #\nRetrieved Context: \n{retrieve_context_lst}\n Useful info:{useful_info}
            pattern = r"'Retrieved Context':\s*\[(.*?)\],"
            content = re.sub(pattern, "[]", content, flags=re.DOTALL)

            content = content.replace("---", "")
            content = content.replace("\n", "\n\n")
            

            index = content.find("- INFO - Event")
            pattern = content[:index]
            pattern =   r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}"+pattern[23:]
            blocks = re.split(pattern, content)
            blocks = [block.strip() for block in blocks if block.strip()]

            with st.chat_message("assistant"):
                st.write("查找记录中--------")
                for block in blocks:
                    st.divider()
                    st.write(block)
    except FileNotFoundError:
        with st.chat_message("assistant"):
            content = f"记录 {show_file} 未找到。"
            st.write(content)
        return
    except Exception as e:
        with st.chat_message("assistant"):
            content = f"读取记录 {show_file} 时出错: {e}"
            st.write(content)
        return