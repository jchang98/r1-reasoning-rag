from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
# from tavily import TavilyClient
from src.client import SearchClient, DataClient
from src.utils import * 
from dotenv import load_dotenv
import os
from src.prompts import Prompts
from src.llm import r1, v3, get_r1_ask
from src.logging import log_event, initial_logger
from src.utils import console
from src.mindmap_graph import *
from src.mindmap_visual import visualize_graphml
import json
import re
from graphviz import Digraph
from datetime import datetime
import itertools
import streamlit as st
from streamlit_mermaid import st_mermaid
from queue import Queue
import asyncio
import uuid
load_dotenv()
now = datetime.now().strftime("%Y-%m-%d")

global MAX_LOOP_COUNT
MAX_LOOP_COUNT = 5
date_format = "%Y-%m-%d %H:%M:%S"

max_log_file_length = 100


import io, sys
import signal
import traceback
import concurrent.futures
def run_python_code(code, timeout_duration=5):
    """异步执行Python代码并设置超时"""
    output = io.StringIO()
    old_stdout = sys.stdout
    
    try:
        sys.stdout = output
        
        # 创建一个新的globals字典，包含当前环境的所有内容
        exec_globals = globals().copy()
        
        # 确保有基本的内置函数
        exec_globals.update({
            '__builtins__': __builtins__,
            '__name__': '__main__',
            '__file__': '__dynamic__',
        })
        
        # 预先导入可能需要的包
        try:
            import pandas as pd
            import numpy as np
            import talib
            import scipy
            import statsmodels
            import arch
            exec_globals['pd'] = pd
            exec_globals['np'] = np
            exec_globals['talib'] = talib
            exec_globals['scipy'] = scipy
            exec_globals['statsmodels'] = statsmodels
            exec_globals['arch'] = arch
        except ImportError:
            pass
            

        try:
            # 使用exec_globals作为执行环境
            exec(code, exec_globals)
            result = output.getvalue()
            return result
        except Exception as e:
            return f"Error: {str(e)}\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout
        output.close()
   
class GraphState(TypedDict):
    question: str
    fc_querys: list
    retrieved_context: list[str]
    router_decision: str
    answer_to_question: str
    missing_information: str
    reasoning: list[str]
    useful_information: list[str]
    loop_count: int

    useful_quote:list[str]
    outline_start_time: str
    search_count: int
    data_agent_count: int


# 添加一个clarify
async def get_feedback(question: str , model: str = "deepseek-r1-ths", start_time: datetime = datetime.now()):
    # 根据用户输入的问题生成澄清

    log_file_name = question[:max_log_file_length]
    initial_logger(logging_path="logs", enable_stdout=False, log_file_name=f"{log_file_name}_{start_time.strftime('%Y%m%d%H%M%S')}")

    prompt = Prompts.CLARIFY.invoke({"question": question, "now": now}).text
    messages = [
        {"role": "user", "content": prompt}
    ]

    if model == "deepseek-r1-ths":
        llm_output = await get_r1_ask(messages)
    elif model == "deepseek-r1-vol":
        llm_output = await r1.chat.completions.create(
            model="deepseek-r1-250120", messages=messages
        )
    elif model == "deepseek-v3-vol":
        llm_output = await v3.chat.completions.create(
            model="deepseek-v3-250324", messages=messages
        )
    
    if model == "deepseek-v3-vol":
        reasoning = ""
    else:
        reasoning = llm_output.choices[0].message.reasoning_content.strip()
    response = llm_output.choices[0].message.content.strip()


    clarify_json = {
        "user input": question,
        "clarify reasoning": reasoning,
        "feedback": response
    }
    # console.print(f"==== CLARIFY ====\n{clarify_json}")
    log_event(f"==== CLARIFY ====\n{clarify_json}")

    with st.chat_message("user").container():
        st.markdown(question)
    with st.chat_message("assistant").container():
        st.markdown(f"==== CLARIFY ====")
        st.write(clarify_json)
    return response


class QAAgent:
    def __init__(self, writing_method: str = "parallel", model: str = "deepseek-r1-ths", writing_model: str = "deepseek-v3-ths", data_eng: str = "ifind_data"):
        self.data_client = DataClient()
        self.tavily_client = SearchClient()
        self.workflow = self.create_workflow(writing_method)
        self.model = model
        self.writing_model = writing_model
        self.data_eng = data_eng
        # self.outline2maxloop = 0


    async def fc(self, question):
        # 根据question生成fc_querys                 
        
        if self.data_eng == "none":
            prompt = Prompts.GEN_FC_QUERY_NO_DATA.invoke({"question": question, "now": now}).text
        else:
            prompt = Prompts.GEN_FC_QUERY.invoke({"question": question, "now": now}).text
        messages = [
            {"role": "user", "content": prompt}
        ]

        if self.model == "deepseek-r1-ths":
            llm_output = await get_r1_ask(messages)
        elif self.model == "deepseek-r1-vol":
            llm_output = await r1.chat.completions.create(
                model="deepseek-r1-250120", messages=messages
            )
        elif self.model == "deepseek-v3-vol":
            llm_output = await v3.chat.completions.create(
                model="deepseek-v3-250324", messages=messages
            )
        
        if self.model == "deepseek-v3-vol":
            reasoning = ""
        else:
            reasoning = llm_output.choices[0].message.reasoning_content.strip()
        response = llm_output.choices[0].message.content.strip()
        try:
            response = eval(re.findall(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)[0])
        except:
            console.print("fc_querys生成解析失败，使用原始question")
            log_event("fc_querys生成解析失败，使用原始question")
            response = [{'name': 'search', 'question': question}]

        return response, reasoning
    
    async def get_useful_info(self, question, retrieved_context):
        # 获得有用的信息
        prompt = Prompts.GEN_USEFUL_INFO.invoke({"retrieved_context": retrieved_context, "question": question, "now": now}).text
        messages = [
            {"role": "user", "content": prompt}
        ]

        if self.model == "deepseek-r1-ths":
            llm_output = await get_r1_ask(messages)
        elif self.model == "deepseek-r1-vol":
            llm_output = await r1.chat.completions.create(
                model="deepseek-r1-250120", messages=messages
            )
        elif self.model == "deepseek-v3-vol":
            llm_output = await v3.chat.completions.create(
                model="deepseek-v3-250324", messages=messages
            )
        
        if self.model == "deepseek-v3-vol":
            reasoning = ""
        else:
            reasoning = llm_output.choices[0].message.reasoning_content.strip()
        response = llm_output.choices[0].message.content.strip()
        return response, reasoning
        
    async def get_data_analyze(self, question, data_info):
        # 进行数据分析
        prompt = Prompts.DATA_ANALYZE.invoke({"data_info": data_info, "question": question, "now": now}).text
        messages = [
            {"role": "user", "content": prompt}
        ]
        if self.model == "deepseek-r1-ths":
            llm_output = await get_r1_ask(messages)
        elif self.model == "deepseek-r1-vol":
            llm_output = await r1.chat.completions.create(
                model="deepseek-r1-250120", messages=messages
            )
        elif self.model == "deepseek-v3-vol":
            llm_output = await v3.chat.completions.create(
                model="deepseek-v3-250324", messages=messages
            )
        if self.model == "deepseek-v3-vol":
            reasoning = ""
        else:
            reasoning = llm_output.choices[0].message.reasoning_content.strip()
        response = llm_output.choices[0].message.content.strip()
        return response, reasoning

    async def retrieve(self, state: GraphState):
        outline_start_time = datetime.now().strftime(date_format)
        search_count = state.get("search_count", 0)
        data_agent_count = state.get("data_agent_count", 0)
        # print("\n=== STEP 1: RETRIEVAL ===")

        # 对原始questions进行fc
        question = state["question"]
        fc_querys, fc_querys_reasoning = await self.fc(question)

        for item in fc_querys:
            if item['name'] == 'search':
                search_count += 1
            elif item['name'] == 'data_agent':
                data_agent_count += 1

        # 定义异步处理函数
        async def process_query(q):
            if q['name'] == "search":
                return await self.tavily_client.search(q["question"])
            else:
                if self.data_eng == "ifind_data":
                    result =  await self.data_client.ifind_data(q["question"])
                    data_info_lst = [self.get_data_analyze(item['content'][:4000], item['query']) for item in result if item.get("type") == "index"]
                    data_info_res = await asyncio.gather(*data_info_lst)
                    data_info_code = [re.findall(r"```(?:python)?\s*(.*?)\s*```", t[0] ,re.DOTALL)[0] for t in data_info_res]

                    data_info_code_res = [run_python_code(code) for code in data_info_code]
                    console.print("=== data info code res ===", data_info_code_res)

                    new_result = []
                    for t, t_data in zip(result, data_info_code_res):
                        if "error" in t_data.lower():
                            continue

                        t['old_content'] = t['content']
                        t['content'] = t_data
                        new_result.append(t)
                    return new_result
                elif self.data_eng == "ifind_data_agent":
                    return await self.data_client.ifind_data_agent(q["question"])

        # 使用异步并行处理
        result = await async_parallel_process(fc_querys, process_query)
        
        result = list(itertools.chain.from_iterable(result))
        retrieved_context = []
        for r in result:
            retrieved_context.append({
                "type": r.get("type", ""),
                "content": r.get('content', '')[:4000],
                "old_content": r.get('old_content', '')[:4000],
                "url": r.get('url', ''),
            })

        # 获得搜索的有用的信息
        retrieved_search_content = ""
        idx = 0
        for r in result:
            if r.get("type") == "search":
                web_page_content = r.get('content', '')[:4000]
                retrieved_search_content += f"<webPage {idx+1} begin>\n{web_page_content}\n\nlink:{r.get('url', '')}\n<webPage {idx+1} end>\n\n\n"
                idx += 1

        search_useful_info, search_useful_info_reasoing = await self.get_useful_info(question, retrieved_search_content)
        #解析得到的{url, useful_information}
        try:
            search_useful_info = eval(re.findall(r"```(?:json)?\s*(.*?)\s*```", search_useful_info, re.DOTALL)[0])
        except:
            console.print("search_useful_info 生成解析失败")
            log_event("search_useful_info 生成解析失败")

        # 把搜索的有用信息 和 数据的进行合并成useful info 
        useful_info = []
        for item in search_useful_info:
            useful_info.append({"url": item["url"], "content": item["useful_information"][:4000]})
        for item in result:
            if item.get("type") == "index":
                useful_info.append({"url": item.get("url", ""), "content": item.get("content", "")[:4000], "old_content": item.get("old_content", "")[:4000]})


        retrieve_json = {
            "Searching for": question,
            "fc_querys_reasoning": fc_querys_reasoning,
            "fc_querys": fc_querys,
            "Retrieved Context": retrieved_context,
            "Useful info reasoing": search_useful_info_reasoing,
            "Useful info": useful_info,
        }
        retrieve_json_brief = retrieve_json.copy()
        retrieve_json_brief["Retrieved Context"] = "..."

        console.print(f"\n=== STEP 1: RETRIEVAL ===\n{retrieve_json_brief}")
        log_event(f"\n=== STEP 1: RETRIEVAL ===\n{retrieve_json}")
        
        with st.chat_message("assistant"):
            st.markdown(f"\n=== STEP 1: RETRIEVAL ===")
            st.write(retrieve_json_brief)

        return {"retrieved_context": retrieved_context, "useful_information": useful_info, "outline_start_time": outline_start_time,"search_count": search_count,  "data_agent_count": data_agent_count}

    async def validate_retrieval(self, state: GraphState):
        # print("\n=== STEP 2: VALIDATION ===")

        question = state["question"]
        retrieved_context = state["retrieved_context"]
        useful_information = state['useful_information']
        reasoning = state.get('reasoning',[])

        # 从state中获取循环计数，如果没有则初始化为1
        loop_count = state.get("loop_count", 0)
        loop_count += 1
        if loop_count > MAX_LOOP_COUNT:  # 设置最大循环次数为MAX_LOOP_COUNT
            console.print("达到最大重试次数，强制完成")
            log_event("达到最大重试次数，强制完成")
            # self.outline2maxloop = self.outline2maxloop + 1
            return {
                "router_decision": "COMPLETE",  # 强制完成
                "retrieved_context": state["retrieved_context"],
                "useful_information": state.get("useful_information",[]),
                "missing_information": state.get("missing_information",""),
                "reasoning": state.get("reasoning",[]),
                "loop_count": loop_count
            }

        useful_information_str = ""
        for idx, r in enumerate(useful_information):
            web_page_content = r.get('content', '')[:4000]
            useful_information_str += f"<webPage {idx+1} begin>\n{web_page_content}\n\nlink:{r.get('url', '')}\n<webPage {idx+1} end>\n\n\n"

        prompt = Prompts.VALIDATE_RETRIEVAL.invoke({"useful_information": useful_information_str, "question": question, "now": now}).text
        messages = [
            {"role": "user", "content": prompt}
        ]

        if self.model == "deepseek-r1-ths":
            llm_output = await get_r1_ask(messages)
        elif self.model == "deepseek-r1-vol":
            llm_output = await r1.chat.completions.create(
                model="deepseek-r1-250120", messages=messages
            )
        elif self.model == "deepseek-v3-vol":
            llm_output = await v3.chat.completions.create(
                model="deepseek-v3-250324", messages=messages
            )
        
        if self.model == "deepseek-v3-vol":
            cur_reasoning = ""
        else:
            cur_reasoning = llm_output.choices[0].message.reasoning_content.strip()
        response = llm_output.choices[0].message.content.strip()
        # print("reasoning:", reasoning)

        reasoning.append(cur_reasoning)
        try:
            strcutured_response = json.loads(response)
        except:
            strcutured_response = re.findall(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)[0]
            strcutured_response = eval(strcutured_response)
        
        router_decision = strcutured_response["status"]
        missing_information = strcutured_response["missing_information"]


        validate_json = {
            "global question": question,
            "loop_count": loop_count,
            "router_decision": router_decision,
            "validate reasoning": cur_reasoning,
            "missing information": missing_information,
        }

        console.print(f"\n=== STEP 2: VALIDATION ===\n{validate_json}")
        log_event(f"\n=== STEP 2: VALIDATION ===\n{validate_json}")
        with st.chat_message("assistant"):
            st.markdown(f"\n=== STEP 2: VALIDATION ===")
            st.write(validate_json)


        return {"router_decision": router_decision, "retrieved_context": retrieved_context, "useful_information": useful_information, "missing_information": missing_information, "reasoning": reasoning, "loop_count": loop_count}

    async def no_answer(self, state: GraphState):
        # print("\n=== STEP 3: NO ANSWER ===")
        question = state["question"]
        context = state["retrieved_context"]
        useful_information = state['useful_information']
        validate_reasoning = state['reasoning']
        answer = ""
    
        console.print(f"\n=== STEP 3: NO ANSWERING ===\nglobal question: {question}\nuseful information:{str(useful_information)}\nfinal_answer: {answer}")
        log_event(f"\n=== STEP 3: NO ANSWERING ===\nglobal question: {question}\nuseful information:{str(useful_information)}\nfinal_answer: {answer}")

        return {"answer_to_question": answer, "retrieved_context": context, "useful_information": state['useful_information'], "question": question, "reasoning": validate_reasoning }

    async def answer(self, state: GraphState):
        # print("\n=== STEP 3: ANSWERING ===")

        question = state["question"]
        context = state["retrieved_context"]
        useful_information = state['useful_information']
        outline_start_time = state['outline_start_time']
        loop_count = state['loop_count']
        search_count = state["search_count"]
        data_agent_count = state["data_agent_count"]
        validate_reasoning = state['reasoning']

        # 根据validate_reasoning + 标题子标题信息生成推理路径，用来指导写作
        validate_reasoning_str = "\n<reasoning end>\n".join(validate_reasoning)
        prompt = Prompts.GEN_REASONING_PATH.invoke({"validate_reasoning": validate_reasoning_str, "question": question, "now": now}).text
        messages = [
            {"role": "user", "content": prompt}
        ]
        if self.model == "deepseek-r1-ths":
            llm_output = await get_r1_ask(messages)
        elif self.model == "deepseek-r1-vol":
            llm_output = await r1.chat.completions.create(
                model="deepseek-r1-250120", messages=messages
            )
        elif self.model == "deepseek-v3-vol":
            llm_output = await v3.chat.completions.create(
                model="deepseek-v3-250324", messages=messages
            )
        
        if self.model == "deepseek-v3-vol":
            reasoning_path_reason = ""
        else:
            reasoning_path_reason = llm_output.choices[0].message.reasoning_content.strip()
        reasoning_path = llm_output.choices[0].message.content.strip() 

        useful_information_str = ""
        for idx, r in enumerate(useful_information):
            web_page_content = r.get('content', '')[:4000]
            useful_information_str += f"<webPage {idx+1} begin>\n{web_page_content}\n\nlink:{r.get('url', '')}\n<webPage {idx+1} end>\n\n\n"

        # 将useful info构建为graph，提取点和边的信息
        mind_map = MindMap(useful_information_str)
        await mind_map.initialize()

        # 读取./local_mem/graph_chunk_entity_relation.graphml 转化为点和边
        entity_context = mind_map.get_entity_csv("./local_mem/graph_chunk_entity_relation.graphml")
        relations_context  = mind_map.get_relation_csv("./local_mem/graph_chunk_entity_relation.graphml")



        prompt = Prompts.ANSWER_QUESTION.invoke({"useful_information": useful_information_str, "question": question, "reasoning_path": reasoning_path, "entity_context":entity_context, "relations_context": relations_context, "now": now}).text
        messages = [
            {"role": "user", "content": prompt}
        ]
        

        if self.writing_model == "deepseek-r1-ths":
            llm_output = await get_r1_ask(messages)
        elif self.writing_model == "deepseek-r1-vol":
            llm_output = await r1.chat.completions.create(
                model="deepseek-r1-250120", messages=messages
            )
        elif self.writing_model == "deepseek-v3-vol":
            llm_output = await v3.chat.completions.create(
                model="deepseek-v3-250324", messages=messages
            )
        
        if self.writing_model == "deepseek-v3-vol":
            reasoning = ""
        else:
            reasoning = llm_output.choices[0].message.reasoning_content.strip()
        answer = llm_output.choices[0].message.content.strip()

        # 删除answer中可能包含的标题
        topic_title = re.findall('请以‘(.*)’为标题', question)[0]
        sec_title = re.findall('，‘(.*)’为子标题', question)[0]
        answer = re.sub(rf"(#*\s*{re.escape(topic_title)}|\*\*\s*{re.escape(topic_title)}\s*\*\*)\s*\n", "", answer).strip()
        answer = re.sub(rf"(#*\s*{re.escape(sec_title)}|\*\*\s*{re.escape(sec_title)}\s*\*\*)\s*\n", "", answer).strip()

        #计算该outline的时间
        parsed_date = datetime.strptime(outline_start_time, date_format)
        outline_time = datetime.now() - parsed_date


        answer_json = {
            "golobal question": question,
            "useful information": useful_information_str,
            "reasoning_path_reason": reasoning_path_reason,
            "reasoning_path": reasoning_path,
            "final_answer_reasoning": reasoning,
            "final_answer": answer,
            "cost time": outline_time,
            "all loop count": loop_count,
            "all search count": search_count,
            "all data_agent_count": data_agent_count,
            "answer length": len(answer)
        }
        console.print(f"\n=== STEP 3: ANSWERING ===\n{answer_json}")
        log_event(f"\n=== STEP 3: ANSWERING ===\n{answer_json}")
        with st.chat_message("assistant"):
            st.markdown(f"\n=== STEP 3: ANSWERING ===")
            st.write(answer_json)


        return {"answer_to_question": answer, "retrieved_context": context, "useful_information": state['useful_information'], "question": question, "reasoning": validate_reasoning }

    async def find_missing_information(self, state: GraphState):
        # print("\n=== STEP 2b: FINDING MISSING INFORMATION ===")

        question = state['question']
        missing_information = state["missing_information"]
        search_count = state["search_count"]
        data_agent_count = state["data_agent_count"]

        # 对missing_information进行fc
        missing_information_query = f"{question}\n\n当前遗漏的信息为：{missing_information}\n\n请根据当前遗漏的信息，生成新的搜索查询问句。"
        fc_querys, fc_querys_reasoning = await self.fc(missing_information_query)
        for item in fc_querys:
            if item['name'] == 'search':
                search_count += 1
            elif item['name'] == 'data_agent':
                data_agent_count += 1

        # 定义异步处理函数
        async def process_query(q):
            if q['name'] == "search":
                return await self.tavily_client.search(q["question"])
            else:
                if self.data_eng == "ifind_data":
                    result =  await self.data_client.ifind_data(q["question"])
                    data_info_lst = [self.get_data_analyze(item['content'][:4000], item['query']) for item in result if item.get("type") == "index"]
                    data_info_res = await asyncio.gather(*data_info_lst)
                    data_info_code = [re.findall(r"```(?:python)?\s*(.*?)\s*```",t[0] ,re.DOTALL)[0] for t in data_info_res]

                    data_info_code_res = [run_python_code(code) for code in data_info_code]
                    console.print("=== data info code res ===", data_info_code_res)

                    new_result = []
                    for t, t_data in zip(result, data_info_code_res):
                        if "error" in t_data.lower():
                            continue

                        t['old_content'] = t['content']
                        t['content'] = t_data
                        new_result.append(t)
                    return new_result
                elif self.data_eng == "ifind_data_agent":
                    return await self.data_client.ifind_data_agent(q["question"])

        # 使用异步并行处理
        new_results = await async_parallel_process(fc_querys, process_query)
        
        new_results = list(itertools.chain.from_iterable(new_results))
        previously_retrieved_useful_information = state["useful_information"]


        # 获得所有的新信息
        new_retrieved_context = []
        for r in new_results:
            new_retrieved_context.append({
                "type": r.get("type", ""),
                "content": r.get('content', '')[:4000],
                "old_content": r.get('old_content', '')[:4000],
                "url": r.get('url', ''),
            })

        # 获得新搜索的有用的信息
        new_retrieved_search_content = ""
        idx = 0
        for r in new_results:
            if r.get("type") == "search":
                web_page_content = r.get('content', '')[:4000]
                new_retrieved_search_content += f"<webPage {idx+1} begin>\n{web_page_content}\n\nlink:{r.get('url', '')}\n<webPage {idx+1} end>\n\n\n"
                idx += 1

        new_search_useful_info, new_search_useful_info_reasoning = await self.get_useful_info(question, new_retrieved_search_content)
        #解析得到的{url, useful_information}
        try:
            new_search_useful_info = eval(re.findall(r"```(?:json)?\s*(.*?)\s*```", new_search_useful_info, re.DOTALL)[0])
        except:
            console.print("new_search_useful_info 生成解析失败")
            log_event("new_search_useful_info 生成解析失败")

        # 把新搜索的有用信息 和 新数据的进行合并成new useful info 
        new_useful_info = []
        for item in new_search_useful_info:
            new_useful_info.append({"url": item["url"], "content": item["useful_information"][:4000]})
        for item in new_results:
            if item.get("type") == "index":
                new_useful_info.append({"url": item.get("url", ""), "content": item.get("content", "")[:4000], "old_content": item.get("old_content", "")[:4000]})


        newly_retrieved_context_lst = str(new_results)
        newly_useful_info = str(new_useful_info)

        miss_infor_json = {
            "global question": question,
            "Searching for missing": missing_information,
            "fc_querys_reasoning": fc_querys_reasoning,
            "fc_querys": fc_querys,
            "Newly retrieved context": new_retrieved_context,
            "Newly useful info reasoning": new_search_useful_info_reasoning,
            "Newly useful info": new_useful_info,
        }
        miss_infor_json_brief = miss_infor_json.copy()
        miss_infor_json_brief['Newly retrieved context'] = '...'

        console.print(f"\n=== STEP 2b: FINDING MISSING INFORMATION ===\n{miss_infor_json_brief}")
        log_event(f"\n=== STEP 2b: FINDING MISSING INFORMATION ===\n{miss_infor_json}")

        with st.chat_message("assistant"):
            st.markdown(f"\n=== STEP 2b: FINDING MISSING INFORMATION ===")
            st.write(miss_infor_json_brief)


        combined_context = previously_retrieved_useful_information + new_useful_info
        # print("newly retrieved context:", newly_retrieved_context)
        return {"useful_information": combined_context,"search_count": search_count,  "data_agent_count": data_agent_count}

    @staticmethod
    def decide_route(state: GraphState):
        return state["router_decision"]

    def create_workflow(self, writing_method: str = "parallel"):
        workflow = StateGraph(GraphState)
        
        workflow.add_node("retrieve context", self.retrieve)
        workflow.add_node("is retrieved context complete?", self.validate_retrieval)
        workflow.add_node("answer", self.answer)
        workflow.add_node("no answer", self.no_answer)
        workflow.add_node("find missing information", self.find_missing_information)
        
        workflow.set_entry_point("retrieve context")
        workflow.add_edge("retrieve context", "is retrieved context complete?")
        
        if set(writing_method) & set(["parallel", "polish"]):
            workflow.add_conditional_edges(
                "is retrieved context complete?",
                self.decide_route,
                {
                    "COMPLETE": "answer",
                    "INCOMPLETE": "find missing information"
                }
            )

            workflow.add_edge("find missing information", "is retrieved context complete?")
    
            workflow.add_edge("answer", END)
            compiled_graph = workflow.compile()
        elif set(writing_method) & set(["serial"]):
            workflow.add_conditional_edges(
                "is retrieved context complete?",
                self.decide_route,
                {
                    "COMPLETE": "no answer",
                    "INCOMPLETE": "find missing information"
                }
            )

        workflow.add_edge("find missing information", "is retrieved context complete?")
    
        workflow.add_edge("no answer", END)
        compiled_graph = workflow.compile()
        # compiled_graph.get_graph(xray=1).draw_mermaid_png(output_file_path="agent-architecture.png")

        return compiled_graph

    async def gen_outline(self, question: str, max_outline_num: int = 8):
        # print("\n=== OUTLINES GENERATION ===")
        prompt = Prompts.OUTLINES_GEN.invoke({"question": question, "max_outline_num": max_outline_num, "now": now}).text
        messages = [
            {"role": "user", "content": prompt}
        ]

        if self.model == "deepseek-r1-ths":
            llm_output = await get_r1_ask(messages)
        elif self.model == "deepseek-r1-vol":
            llm_output = await r1.chat.completions.create(
                model="deepseek-r1-250120", messages=messages
            )
        elif self.model == "deepseek-v3-vol":
            llm_output = await v3.chat.completions.create(
                model="deepseek-v3-250324", messages=messages
            )
        
        if self.model == "deepseek-v3-vol":
            reasoning = ""
        else:
            reasoning = llm_output.choices[0].message.reasoning_content.strip()
        answer = llm_output.choices[0].message.content.strip()
        answer = eval(re.findall(r"```(?:json)?\s*(.*?)\s*```", answer, re.DOTALL)[0])


        outline_json = {
            "outlines reasoning": reasoning,
            "Init Outlines": answer
        }
        console.print(f"=== Outlines ====\n{outline_json}")
        log_event(f"=== Outlines ====\n{outline_json}")
        return answer, reasoning
    
    def get_final_report(self, outlines: list, results: list):
        assert len(outlines) == len(results)
        final_report = ''
        for outline, result in zip(outlines, results):
            outline_name = re.findall('，‘(.*)’为子标题', outline)[0]
            final_report += f"# {outline_name}\n{result['answer_to_question']}\n\n"
            # final_report += f"{result['answer_to_question']}\n\n"

        return final_report

    async def polish(self, question: str, outlines: list, report: list, polish_step: int = -1):
        # print("\n=== POLISHING ===")
        if polish_step > 0 and polish_step < len(outlines):
            split_outlines = [outlines[i:i + polish_step] for i in range(0, len(outlines), polish_step)]
            split_reports = [report[i:i + polish_step] for i in range(0, len(report), polish_step)]
        else:
            split_outlines = [outlines]
            split_reports = [report]

        # 将两层列表展平，第一层为需要整体润色的多个
        polish_prompt_lsts = []
        for polish_outline, polish_report in zip(split_outlines, split_reports):
            # 拼接大纲和报告
            polish_outline_str = "\n".join([f"# {t['headings']}\n - {t['research_goal']}" for t in polish_outline])

            polish_report_str = "\n\n".join([f"# {t_outline['headings']}\n{t_report['answer_to_question']}" for t_outline, t_report in zip(polish_outline, polish_report)])

            prompt = Prompts.POLIESH_ANSWER.invoke({"question": question, "outlines": polish_outline_str, "draft_writing": polish_report_str, "now": now}).text
            messages = [
                {"role": "user", "content": prompt}
            ]

            polish_prompt_lsts.append(messages)

        # 定义异步处理函数
        async def process_polish(messages):
            if self.writing_model == "deepseek-r1-ths":
                return await get_r1_ask(messages)
            elif self.writing_model == "deepseek-r1-vol":
                return await r1.chat.completions.create(
                    model="deepseek-r1-250120", messages=messages
                )
            elif self.writing_model == "deepseek-v3-vol":
                return await v3.chat.completions.create(
                    model="deepseek-v3-250324", messages=messages
                )

        
        # 使用异步并行处理
        result = await async_parallel_process(polish_prompt_lsts, process_polish)


        # 将所有result中的结果拼接起来
        polish_final_result = []
        polish_final_reasoning = []
        for r in result:
            t_reasoning = r.choices[0].message.reasoning_content.strip()
            t_answer = r.choices[0].message.content.strip()
            polish_final_result.append(t_answer)
            polish_final_reasoning.append(t_reasoning)

        polish_final_result = "\n\n".join(polish_final_result)

        polish_json = {
            "polish report reasoning": str(polish_final_reasoning),
            "polish report": polish_final_result
        }
        console.print(f"==== POLISH REPORT ====\n{polish_json}")
        log_event(f"==== POLISH REPORT ====\n{polish_json}")
        return polish_final_result
    
    async def serial(self, question: str, outlines: list, reports: list):
        # print("\n=== serial ===")

        # 串行生成
        already_writing = ""
        serial_results_lsts = []
        for outline, report in zip(outlines, reports):

            useful_information = report['useful_information']
            sec_question = report['question']
            validate_reasoning = report['reasoning']
            # 根据validate_reasoning + 标题子标题信息生成推理路径，用来指导写作
            validate_reasoning_str = "\n<reasoning end>\n".join(validate_reasoning)
            
            prompt = Prompts.GEN_REASONING_PATH.invoke({"validate_reasoning": validate_reasoning_str, "question": question, "now": now}).text
            messages = [
                {"role": "user", "content": prompt}
            ]
            if self.model == "deepseek-r1-ths":
                llm_output = await get_r1_ask(messages)
            elif self.model == "deepseek-r1-vol":
                llm_output = await r1.chat.completions.create(
                    model="deepseek-r1-250120", messages=messages
                )
            elif self.model == "deepseek-v3-vol":
                llm_output = await v3.chat.completions.create(
                    model="deepseek-v3-250324", messages=messages
                )

            if self.model == "deepseek-v3-vol":
                reasoning_path_reason = ""
            else:
                reasoning_path_reason = llm_output.choices[0].message.reasoning_content.strip()
            reasoning_path = llm_output.choices[0].message.content.strip() 

            useful_information_str = ""
            for idx, r in enumerate(useful_information):
                web_page_content = r.get('content', '')[:4000]
                useful_information_str += f"<webPage {idx+1} begin>\n{web_page_content}\n\nlink:{r.get('url', '')}\n<webPage {idx+1} end>\n\n\n"

            # 将useful info构建为graph，提取点和边的信息
            mind_map = MindMap(useful_information_str)
            await mind_map.initialize()

            # 读取./local_mem/graph_chunk_entity_relation.graphml 转化为点和边
            entity_context = mind_map.get_entity_csv("./local_mem/graph_chunk_entity_relation.graphml")
            relations_context  = mind_map.get_relation_csv("./local_mem/graph_chunk_entity_relation.graphml")

            prompt = Prompts.ANSWER_QUESTION_SERIAL.invoke({"useful_information": useful_information_str, "question": sec_question, "already_writing": already_writing, "reasoning_path": reasoning_path, "entity_context":entity_context, "relations_context": relations_context, "now": now}).text
            messages = [
                {"role": "user", "content": prompt}
            ]

            if self.writing_model == "deepseek-r1-ths":
                llm_output = await get_r1_ask(messages)
            elif self.writing_model == "deepseek-r1-vol":
                llm_output = await r1.chat.completions.create(
                    model="deepseek-r1-250120", messages=messages
                )
            elif self.writing_model == "deepseek-v3-vol":
                llm_output = await v3.chat.completions.create(
                    model="deepseek-v3-250324", messages=messages
                )

            if self.writing_model == "deepseek-v3-vol":
                t_reasoning = ""
            else:
                t_reasoning = llm_output.choices[0].message.reasoning_content.strip()
            t_answer = llm_output.choices[0].message.content.strip()

            # 删除t_answer中可能包含的标题
            topic_title = re.findall('请以‘(.*)’为标题', sec_question)[0]
            sec_title = re.findall('，‘(.*)’为子标题', sec_question)[0]
            t_answer = re.sub(rf"(#*\s*{re.escape(topic_title)}|\*\*\s*{re.escape(topic_title)}\s*\*\*)\s*\n", "", t_answer).strip()
            t_answer = re.sub(rf"(#*\s*{re.escape(sec_title)}|\*\*\s*{re.escape(sec_title)}\s*\*\*)\s*\n", "", t_answer).strip()

            serial_results_lsts.append(t_answer)
            already_writing = already_writing + f"# {outline['headings']}\n{t_answer}\n\n"
            sec_serial_final_result = f"# {outline['headings']}\n{t_answer}\n\n"

            # already_writing = already_writing + f"{t_answer}\n\n"
            # sec_serial_final_result = t_answer

            t_serial_json = {
                "reasoning_path_reason": reasoning_path_reason,
                "reasoning_path": reasoning_path,
                "useful info": useful_information_str,
                "section serial report reasoning": t_reasoning,
                "section serial report": sec_serial_final_result
            }
            with st.chat_message("assistant"):
                st.markdown(f"==== SERIAL REPORT CONTINUE ====")
                st.write(t_serial_json)


            console.print(f"==== SERIAL REPORT CONTINUE ====\n{t_serial_json}")
            log_event(f"==== SERIAL REPORT CONTINUE ====\n{t_serial_json}")
        return already_writing


    def mermaid(self, report: str, add_str: str = ""):
        with st.chat_message("assistant"):
            st.markdown(f"==== FINAL REPORT ({add_str}) ====")
            if "```mermaid" in report:
                parts = re.split(r'(```mermaid.*?```)', report, flags=re.DOTALL)
                for part in parts:
                    if not part.strip():
                        continue
                    if part.strip().startswith('```mermaid'):
                        code = re.search(r'```mermaid\s*(.*?)\s*```', part, re.DOTALL).group(1)
                        st_mermaid(code, key=str(uuid.uuid4()))
                    else:
                        st.markdown(part)

            else:
                st.markdown(report)

    
    async def conclusion(self, question: str, report: str):
        # print("\n=== CONCLUSION ===")
        prompt = Prompts.CONCLUSION.invoke({"question": question, "report": report, "now": now}).text
        messages = [
            {"role": "user", "content": prompt}
        ]

        if self.writing_model == "deepseek-r1-ths":
            llm_output = await get_r1_ask(messages)
        elif self.writing_model == "deepseek-r1-vol":
            llm_output = await r1.chat.completions.create(
                model="deepseek-r1-250120", messages=messages
            )
        elif self.writing_model == "deepseek-v3-vol":
            llm_output = await v3.chat.completions.create(
                model="deepseek-v3-250324", messages=messages
            )

        if self.writing_model == "deepseek-v3-vol":
            reasoning = ""
        else:
            reasoning = llm_output.choices[0].message.reasoning_content.strip()
        answer = llm_output.choices[0].message.content.strip()


        conclusion_json = {
            "conclusion reasoning": reasoning,
            "conclusion": answer
        }
        console.print(f"\n==== CONCLUSION  ====\n{conclusion_json}")
        log_event(f"\n==== CONCLUSION  ====\n{conclusion_json}")
        return answer, reasoning


    async def run(self, question: str, writing_method: str = "parallel", polish_step: int = -1, max_outline_num: int = 8, max_loop: int = 5, feedback: str = "", feedback_answer: str = "", start_time: datetime = datetime.now()):     
        # 先生成一个大纲，然后每个大纲去调用 workflow.invoke
        global MAX_LOOP_COUNT
        MAX_LOOP_COUNT = max_loop

        # self.outline2maxloop = 0
        # Combine information
        combined_query = f"""
        Initial Query: {question}
        Clarify Questions and Answers:
        Q: {feedback}
        A: {feedback_answer}
        """

        with st.chat_message("assistant"):
            st.markdown(f"==== CLARIFY ANSWER ====")
            st.write(combined_query)

        outlines, outlines_reasoning = await self.gen_outline(combined_query, max_outline_num)

        outlines_json = {
            "outlines reasoning": outlines_reasoning,
            "Init Outlines": outlines
        }
        with st.chat_message("assistant"):
            st.markdown(f"==== OUTLINES  ====")
            st.write(outlines_json)


        outlines_lst = [f"请以‘{question}’为标题，‘{item['headings']}’为子标题，展开深度研究，研究目标为‘{item['research_goal']}’" for item in outlines]

        with st.chat_message("assistant"):
            st.markdown(f"==== 并行查找资料中，请耐心等待 ====")
        # 定义异步处理函数
        async def process_outline(outline):
            return await self.workflow.ainvoke({"question": f"{outline}"}, {"recursion_limit": 999})
        
        # 使用异步并行处理
        results = await async_parallel_process(outlines_lst, process_outline)

        outline2maxloop = len([res for res in results if res.get("loop_count") > MAX_LOOP_COUNT])
        with st.chat_message("assistant"):
            st.markdown(f"{outline2maxloop}个outline达到了最大循环次数")
            log_event(f"{outline2maxloop}个outline达到了最大循环次数")


        # 并行生成
        if set(writing_method) & set(["parallel", "polish"]):
            # 根据result的内容去拼接正文
            parallel_final_report = self.get_final_report(outlines_lst, results)
            console.print(f"\n==== FINAL REPORT (POLISH BEFORE) ====\n{parallel_final_report}")
            log_event(f"\n==== FINAL REPORT (POLISH BEFORE) ====\n{parallel_final_report}")
            # self.mermaid(final_report, add_str="POLISH BEFORE")

            # 进行polish
            polish_final_report = ""
            if "polish" in set(writing_method):
                polish_final_report = await self.polish(question, outlines, results, polish_step=polish_step)
                # self.mermaid(final_report, add_str="POLISH AFTER")
            

            # 添加一个结论
            conclusion, conclusion_reasoing = await self.conclusion(question, parallel_final_report)
            conclusion_json = {
                "conclusion reasoning": conclusion_reasoing,
                "conclusion": conclusion
            }
            with st.chat_message("assistant"):
                st.markdown(f"==== CONCLUSION ====")
                st.write(conclusion_json)

            parallel_final_report = parallel_final_report + f"\n\n{conclusion}"

        if "serial" in set(writing_method):
            serial_final_report = await self.serial(question, outlines, results)

            conclusion, conclusion_reasoing = await self.conclusion(question, serial_final_report)
            conclusion_json = {
                "conclusion reasoning": conclusion_reasoing,
                "conclusion": conclusion
            }
            with st.chat_message("assistant"):
                st.markdown(f"==== CONCLUSION ====")
                st.write(conclusion_json)
            serial_final_report = serial_final_report + f"\n\n{conclusion}"
            # self.mermaid(final_report, add_str="SERIAL")
        

        #进行展示
        if  "polish" in set(writing_method):
            self.mermaid(parallel_final_report, add_str="POLISH BEFORE")
            self.mermaid(polish_final_report, add_str="POLISH AFTER")
        elif "parallel" in set(writing_method):
            self.mermaid(parallel_final_report, add_str="POLISH BEFORE")
        if  "serial" in set(writing_method):
            self.mermaid(serial_final_report, add_str="SERIAL")

        # 展示mindmap
        graphml_file = "./local_mem/graph_chunk_entity_relation.graphml"
        html_file = "./local_mem/graph_chunk_entity_relation.html"
        visualize_graphml(graphml_file, html_file)

        with open(html_file, 'r', encoding='utf-8') as file:
            html_content = file.read()
        with st.chat_message("assistant"):
            st.markdown(f"==== MINDMAP ====\n")
            st.components.v1.html(html_content, height=600) 

        end_time = datetime.now()
        print(f"Total time: {end_time - start_time}")
        st.success(f"Generation takes {end_time - start_time} seconds.")
        # Save report
        os.makedirs("output", exist_ok=True)
        log_file_name = question[:max_log_file_length]
        with open(f"output/{log_file_name}_{start_time.strftime('%Y%m%d%H%M%S')}.md", "w") as f:
            if "parallel" in set(writing_method):
                f.write("=== parallel report ===")
                f.write(parallel_final_report)

            if  "polish" in set(writing_method):
                f.write("=== polish report ===")
                f.write(polish_final_report)
            

            if  "serial" in set(writing_method):
                f.write("=== serial report ===")
                f.write(serial_final_report)
        

        # 读取搜索的日志信息
        # with open(f"logs/{question}_{start_time.strftime('%Y%m%d%H%M%S')}.log", "r") as f:
        #     log_content = f.read()
        # with st.expander("ALL Logs:"):
        #     log_content = log_content.replace("\n", "\n\n")
        #     st.markdown(log_content)



if __name__ == "__main__":
    agent = QAAgent()
    agent.run("Who is George Washington?")