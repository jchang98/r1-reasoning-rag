from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from rich.console import Console
import asyncio


console = Console()

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

