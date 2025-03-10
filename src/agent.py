from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
# from tavily import TavilyClient
from src.client import SearchClient, DataClient
from src.utils import * 
from dotenv import load_dotenv
import os
from src.prompts import Prompts
from src.llm import r1
from src.logging import log_event, initial_logger
from src.utils import console
import json
import re
from graphviz import Digraph
from datetime import datetime
import itertools
import streamlit as st
from streamlit_mermaid import st_mermaid
from queue import Queue
load_dotenv()

global MAX_LOOP_COUNT
MAX_LOOP_COUNT = 5


   
class GraphState(TypedDict):
    question: str
    fc_querys: list
    retrieved_context: str
    router_decision: str
    answer_to_question: str
    missing_information: str
    reasoning: str
    useful_information: str
    loop_count: int


class QAAgent:
    def __init__(self):
        self.data_client = DataClient()
        self.tavily_client = SearchClient()
        self.workflow = self.create_workflow()

    def fc(self, question):
        # 根据question生成fc_querys                 
        
        prompt = Prompts.GEN_FC_QUERY.invoke({"question": question}).text
        messages = [
            {"role": "user", "content": prompt}
        ]
        llm_output = r1.chat.completions.create(
            model="ep-20250208165153-wn9ft", messages=messages
        )
        
        reasoning = llm_output.choices[0].message.reasoning_content.strip()
        response = llm_output.choices[0].message.content.strip()
        try:
            response = eval(re.findall(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)[0])
        except:
            console.print("fc_querys生成解析失败，使用原始question")
            log_event("fc_querys生成解析失败，使用原始question")
            response = [{'name': 'search', 'question': question}]

        return response
    
    def get_useful_info(self, question, retrieved_context):
        # 获得有用的信息
        prompt = Prompts.GEN_USEFUL_INFO.invoke({"retrieved_context": retrieved_context, "question": question}).text
        messages = [
            {"role": "user", "content": prompt}
        ]
        llm_output = r1.chat.completions.create(
            model="ep-20250208165153-wn9ft", messages=messages
        )
        reasoning = llm_output.choices[0].message.reasoning_content.strip()
        response = llm_output.choices[0].message.content.strip()
        return response
        


    def retrieve(self, state: GraphState):
        # print("\n=== STEP 1: RETRIEVAL ===")

        # 对原始questions进行fc
        question = state["question"]
        fc_querys = self.fc(question)

        result = parallel_process(
            fc_querys,
            lambda q: self.tavily_client.search(q["question"]) if q['name'] == "search" else self.data_client.ifind_data(q['question'])
        )
        
        result = list(itertools.chain.from_iterable(result))
        retrieved_context = "\n---\n".join([r.get("content","") for r in result])
        retrieve_context_lst = str([r.get("content","") for r in result])

        # 获得有用的信息
        retrieved_search_content = "\n---\n".join([r.get("content","")[:4000] for r in result if r.get("type") == "search"])
        useful_info = self.get_useful_info(question, retrieved_search_content)
        useful_info = useful_info + "\n\n" + "\n---\n".join([r.get("content","")[:4000] for r in result if r.get("type") == "index"])


        console.print(f"\n=== STEP 1: RETRIEVAL ===\nSearching for: {question}\nfc_query:{fc_querys}\nRetrieved Context: \n...\nUseful info: {useful_info}")
        log_event(f"\n=== STEP 1: RETRIEVAL ===\nSearching for: {question}\nfc_query:{fc_querys}\nRetrieved Context: \n{retrieve_context_lst}\n Useful info:{useful_info}")
        
        # with st.chat_message("assistant"):
        #     st.markdown(f"\n=== STEP 1: RETRIEVAL ===\nSearching for: {question}\nfc_query:{fc_querys}\nRetrieved Context: \n...\nUseful info: {str([useful_info])}")

        if hasattr(self, 'message_callback'):
            self.message_callback(f"\n=== STEP 1: RETRIEVAL ===\nSearching for: {question}\nfc_query:{fc_querys}\nRetrieved Context: \n...\nUseful info: {useful_info}")
        return {"retrieved_context": retrieved_context, "useful_information": useful_info}

    def validate_retrieval(self, state: GraphState):
        # print("\n=== STEP 2: VALIDATION ===")

        question = state["question"]
        retrieved_context = state["retrieved_context"]
        useful_information = state['useful_information']

        # 从state中获取循环计数，如果没有则初始化为1
        loop_count = state.get("loop_count", 1)
        if loop_count > MAX_LOOP_COUNT:  # 设置最大循环次数为MAX_LOOP_COUNT
            console.print("达到最大重试次数，强制完成")
            log_event("达到最大重试次数，强制完成")
            return {
                "router_decision": "COMPLETE",  # 强制完成
                "retrieved_context": state["retrieved_context"],
                "useful_information": state.get("useful_information",""),
                "missing_information": state.get("missing_information",""),
                "reasoning": state.get("reasoning",""),
                "loop_count": loop_count+1
            }


        prompt = Prompts.VALIDATE_RETRIEVAL.invoke({"useful_information": useful_information, "question": question}).text
        messages = [
            {"role": "user", "content": prompt}
        ]
        llm_output = r1.chat.completions.create(
            model="ep-20250208165153-wn9ft", messages=messages
        )
        
        reasoning = llm_output.choices[0].message.reasoning_content.strip()
        response = llm_output.choices[0].message.content.strip()
        # print("reasoning:", reasoning)
        try:
            strcutured_response = json.loads(response)
        except:
            strcutured_response = re.findall(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)[0]
            strcutured_response = eval(strcutured_response)
        
        router_decision = strcutured_response["status"]
        missing_information = strcutured_response["missing_information"]

        console.print(f"\n=== STEP 2: VALIDATION ===\nglobal question: {question}\nloop_count:{loop_count}\nrouter decision:{router_decision}\nmissing information:{missing_information}")
        log_event(f"\n=== STEP 2: VALIDATION ===\nglobal question: {question}\nloop_count:{loop_count}\nrouter decision:{router_decision}\nmissing information:{missing_information}")
        # with st.chat_message("assistant"):
        #     st.markdown(f"\n=== STEP 2: VALIDATION ===\nglobal question: {question}\nloop_count:{loop_count}\nrouter decision:{router_decision}\nmissing information:{missing_information}")
        if hasattr(self, 'message_callback'):
            self.message_callback(f"\n=== STEP 2: VALIDATION ===\nglobal question: {question}\nloop_count:{loop_count}\nrouter decision:{router_decision}\nmissing information:{missing_information}")


        return {"router_decision": router_decision, "retrieved_context": retrieved_context, "useful_information": useful_information, "missing_information": missing_information, "reasoning": reasoning, "loop_count": loop_count+1}

    def answer(self, state: GraphState):
        # print("\n=== STEP 3: ANSWERING ===")

        question = state["question"]
        context = state["retrieved_context"]
        useful_information = state['useful_information']

        prompt = Prompts.ANSWER_QUESTION.invoke({"useful_information": useful_information, "question": question}).text
        messages = [
            {"role": "user", "content": prompt}
        ]
        llm_output = r1.chat.completions.create(
            model="ep-20250208165153-wn9ft", messages=messages
        )

        reasoning = llm_output.choices[0].message.reasoning_content.strip()
        answer = llm_output.choices[0].message.content.strip()
        console.print(f"\n=== STEP 3: ANSWERING ===\nglobal question: {question}\nuseful information:{useful_information}\nfinal_answer: {answer}")
        log_event(f"\n=== STEP 3: ANSWERING ===\nglobal question: {question}\nuseful information:{useful_information}\nfinal_answer: {answer}")
        # with st.chat_message("assistant"):
        #     st.markdown(f"\n=== STEP 3: ANSWERING ===\nglobal question: {question}\nfinal_answer: {answer}")
        
        if hasattr(self, 'message_callback'):
            self.message_callback(f"\n=== STEP 3: ANSWERING ===\nglobal question: {question}\nfinal_answer: {answer}")

        return {"answer_to_question": answer, "retrieved_context": context, "useful_information": state['useful_information'] }

    def find_missing_information(self, state: GraphState):
        # print("\n=== STEP 2b: FINDING MISSING INFORMATION ===")

        question = state['question']
        missing_information = state["missing_information"]

        # 对missing_information进行fc
        fc_querys = self.fc(missing_information)

        tavily_query = parallel_process(
            fc_querys,
            lambda q: self.tavily_client.search(q["question"]) if q['name'] == "search" else self.data_client.ifind_data(q['question'])
        )
        
        tavily_query = list(itertools.chain.from_iterable(tavily_query))
        previously_retrieved_useful_information = state["useful_information"]

        newly_retrieved_context = "\n---\n".join([r.get("content","") for r in tavily_query])
        newly_retrieved_context_lst = str([r.get("content","") for r in tavily_query])

        # 获得新的有用的信息
        newly_retrieved_search_content = "\n---\n".join([r.get("content","")[:4000] for r in tavily_query if r.get("type") == "search"])
        newly_useful_info = self.get_useful_info(question, newly_retrieved_search_content)
        newly_useful_info = newly_useful_info + "\n\n" + "\n---\n".join([r.get("content","")[:4000] for r in tavily_query if r.get("type") == "index"])


        console.print(f"\n=== STEP 2b: FINDING MISSING INFORMATION ===\nglobal question: {question}\nSearching for missing:{missing_information}\nfc_querys:{fc_querys}\nNewly retrieved context: \n...\nNewly useful info: {newly_useful_info}")
        log_event(f"\n=== STEP 2b: FINDING MISSING INFORMATION ===\nglobal question: {question}\nSearching for missing:{missing_information}\nfc_querys:{fc_querys}\nNewly retrieved context: \n{newly_retrieved_context_lst}\nNewly useful info: {newly_useful_info}")
        # with st.chat_message("assistant"):
        #     st.markdown(f"\n=== STEP 2b: FINDING MISSING INFORMATION ===\nglobal question: {question}\nSearching for missing:{missing_information}\nfc_querys:{fc_querys}\nNewly retrieved context: \n...\nNewly useful info: {str([newly_useful_info])}")

        if hasattr(self, 'message_callback'):
            self.message_callback(f"\n=== STEP 2b: FINDING MISSING INFORMATION ===\nglobal question: {question}\nSearching for missing:{missing_information}\nfc_querys:{fc_querys}\nNewly retrieved context: \n...\nNewly useful info: {str([newly_useful_info])}")

        combined_context = f"{previously_retrieved_useful_information}\n{newly_useful_info}"
        # print("newly retrieved context:", newly_retrieved_context)
        return {"useful_information": combined_context}

    @staticmethod
    def decide_route(state: GraphState):
        return state["router_decision"]

    def create_workflow(self):
        workflow = StateGraph(GraphState)
        
        workflow.add_node("retrieve context", self.retrieve)
        workflow.add_node("is retrieved context complete?", self.validate_retrieval)
        workflow.add_node("answer", self.answer)
        workflow.add_node("find missing information", self.find_missing_information)
        
        workflow.set_entry_point("retrieve context")
        workflow.add_edge("retrieve context", "is retrieved context complete?")
        
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
        # compiled_graph.get_graph(xray=1).draw_mermaid_png(output_file_path="agent-architecture.png")


        # 使用 graphviz 生成图片
        dot = Digraph(comment='Workflow')
        
        # 添加节点
        dot.node('retrieve context', 'retrieve context')
        dot.node('is retrieved context complete?', 'is retrieved context complete?')
        dot.node('answer', 'answer')
        dot.node('find missing information', 'find missing information')
        
        # 添加边
        dot.edge('retrieve context', 'is retrieved context complete?')
        dot.edge('is retrieved context complete?', 'answer', label='COMPLETE')
        dot.edge('is retrieved context complete?', 'find missing information', label='INCOMPLETE')
        dot.edge('find missing information', 'is retrieved context complete?')
        dot.edge('answer', 'END')
        
        # 生成图片
        dot.render('agent-architecture', format='png', cleanup=True)
        
        return compiled_graph

    def gen_outline(self, question: str, max_outline_num: int = 8):
        # print("\n=== OUTLINES GENERATION ===")
        prompt = Prompts.OUTLINES_GEN.invoke({"question": question, "max_outline_num": max_outline_num}).text
        messages = [
            {"role": "user", "content": prompt}
        ]
        llm_output = r1.chat.completions.create(
            model="ep-20250208165153-wn9ft", messages=messages
        )
        reasoning = llm_output.choices[0].message.reasoning_content.strip()
        answer = llm_output.choices[0].message.content.strip()
        answer = eval(re.findall(r"```(?:json)?\s*(.*?)\s*```", answer, re.DOTALL)[0])

        console.print(f"\nInit Outlines:\n{str(answer)}")
        log_event(f"\nInit Outlines:\n{str(answer)}")
        return answer
    
    def get_final_report(self, outlines: list, results: list):
        assert len(outlines) == len(results)
        final_report = ''
        for outline, result in zip(outlines, results):
            outline_name = re.findall('，‘(.*)’为子标题', outline)[0]
            final_report += f"# {outline_name}\n{result['answer_to_question']}\n\n"

        return final_report

    def polish(self, question: str, outlines: list, report: list, polish_step: int = -1):
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

            prompt = Prompts.POLIESH_ANSWER.invoke({"question": question, "outlines": polish_outline_str, "draft_writing": polish_report_str}).text
            messages = [
                {"role": "user", "content": prompt}
            ]

            polish_prompt_lsts.append(messages)

        result = parallel_process(
            polish_prompt_lsts,
            lambda x: r1.chat.completions.create(
                model="ep-20250208165153-wn9ft", messages=x
            )
        )


        # 将所有result中的结果拼接起来
        polish_final_result = []
        for r in result:
            t_reasoning = r.choices[0].message.reasoning_content.strip()
            t_answer = r.choices[0].message.content.strip()
            polish_final_result.append(t_answer)

        polish_final_result = "\n\n".join(polish_final_result)
        console.print(f"\n==== POLISH REPORT ====\n{polish_final_result}")
        log_event(f"\n==== POLISH REPORT  ====\n{polish_final_result}")
        return polish_final_result


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
                        st_mermaid(code)
                    else:
                        st.markdown(part)

            else:
                st.markdown(report)


    def run(self, question: str, polish: bool = False, polish_step: int = -1, max_outline_num: int = 8, max_loop: int = 5):     
        # 先生成一个大纲，然后每个大纲去调用 workflow.invoke
        global MAX_LOOP_COUNT
        MAX_LOOP_COUNT = max_loop

        from multiprocessing import Manager
        manager = Manager()
        shared_queue = manager.list()
        self.shared_queue = shared_queue

        
        def display_messages():
            for m in self.shared_queue:
                with st.chat_message("assistant"):
                    st.text(m)


        def message_callback(msg):
            self.shared_queue.append(msg)

        self.message_callback = message_callback

        start_time = datetime.now()
        initial_logger(logging_path="logs", enable_stdout=False, log_file_name=f"{question}_{start_time.strftime('%Y%m%d%H%M%S')}")

        outlines = self.gen_outline(question, max_outline_num)
        with st.chat_message("assistant"):
            st.markdown(f"Init Outlines:\n{str(outlines)}")
        outlines_lst = [f"请以‘{question}’为标题，‘{item['headings']}’为子标题，展开深度研究，研究目标为‘{item['research_goal']}’" for item in outlines]

        with st.chat_message("assistant"):
            st.markdown(f"==== 并行查找资料中，请耐心等待 ====")
        results = parallel_process(
            outlines_lst,
            lambda x: self.workflow.invoke({"question": f"{x}"}, {"recursion_limit": 999}),
        )
        # result = self.workflow.invoke({"question": question})
        # 更新并显示消息
        display_messages()

        # 根据result的内容去拼接正文
        final_report = self.get_final_report(outlines_lst, results)
        console.print(f"\n==== FINAL REPORT (POLISH BEFORE) ====\n{final_report}")
        log_event(f"\n==== FINAL REPORT (POLISH BEFORE) ====\n{final_report}")
        # 将final_report中的 ```mermaid`` 替换为 html标签
        self.mermaid(final_report, add_str="POLISH BEFORE")
        # with st.chat_message("assistant"):
        #     st.components.v1.html(f"==== FINAL REPORT (POLISH BEFORE) ====\n{final_report}")


        # 进行polish
        if polish:
            final_report = self.polish(question, outlines, results, polish_step=polish_step)
            self.mermaid(final_report, add_str="POLISH AFTER")
            # with st.chat_message("assistant"):
            #     st.components.v1.html(f"==== POLISH REPORT  ====\n{final_report}")

        end_time = datetime.now()
        print(f"Total time: {end_time - start_time}")
        st.success(f"Generation takes {end_time - start_time} seconds.")
        # Save report
        os.makedirs("output", exist_ok=True)
        with open(f"output/{question}_{start_time.strftime('%Y%m%d%H%M%S')}.md", "w") as f:
            f.write(final_report)
        

        # 读取搜索的日志信息
        # with open(f"logs/{question}_{start_time.strftime('%Y%m%d%H%M%S')}.log", "r") as f:
        #     log_content = f.read()
        # with st.expander("ALL Logs:"):
        #     log_content = log_content.replace("\n", "\n\n")
        #     st.markdown(log_content)

        return final_report

if __name__ == "__main__":
    agent = QAAgent()
    agent.run("Who is George Washington?")