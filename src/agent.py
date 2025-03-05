from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
# from tavily import TavilyClient
from src.client import SearchClient
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
load_dotenv()

class GraphState(TypedDict):
    question: str
    retrieved_context: str
    router_decision: str
    answer_to_question: str
    missing_information: str
    reasoning: str
    useful_information: str

class QAAgent:
    def __init__(self):
        self.tavily_client = SearchClient()
        self.workflow = self.create_workflow()

    def retrieve(self, state: GraphState):
        # print("\n=== STEP 1: RETRIEVAL ===")
        question = state["question"]
        result = self.tavily_client.search(question, max_results=3)
        retrieved_context = "\n".join([r["content"] for r in result])

        console.print(f"\n=== STEP 1: RETRIEVAL ===\nSearching for: {question}\n\nRetrieved Context: \n{retrieved_context}")
        log_event(f"\n=== STEP 1: RETRIEVAL ===\nSearching for: {question}\n\nRetrieved Context: \n{retrieved_context}")
        return {"retrieved_context": retrieved_context}

    def validate_retrieval(self, state: GraphState):
        # print("\n=== STEP 2: VALIDATION ===")
        question = state["question"]
        retrieved_context = state["retrieved_context"]
        # print("Retrieved Context: \n", retrieved_context)
        # validation_chain = Prompts.VALIDATE_RETRIEVAL | r1
        # llm_output = validation_chain.invoke({"retrieved_context": retrieved_context, "question": question}).content

        # 从state中获取循环计数，如果没有则初始化为1
        loop_count = state.get("loop_count", 1)
        if loop_count >= 5:  # 设置最大循环次数为5
            console.print("达到最大重试次数，强制完成")
            log.event("达到最大重试次数，强制完成")
            return {
                "router_decision": "COMPLETE",  # 强制完成
                "retrieved_context": state["retrieved_context"],
                "useful_information": state["useful_information"],
                "missing_information": state["missing_information"],
                "reasoning": state["reasoning"],
                "loop_count": loop_count+1
            }


        prompt = Prompts.VALIDATE_RETRIEVAL.invoke({"retrieved_context": retrieved_context, "question": question}).text
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
            strcutured_response = json.loads(strcutured_response)
        
        router_decision = strcutured_response["status"]
        missing_information = strcutured_response["missing_information"]
        useful_information = strcutured_response["useful_information"]
        console.print(f"\n=== STEP 2: VALIDATION ===\nrouter decision:{router_decision}\nmissing information:{missing_information}\nuseful information:{useful_information}")
        log_event(f"\n=== STEP 2: VALIDATION ===\nrouter decision:{router_decision}\nmissing information:{missing_information}\nuseful information:{useful_information}")

        if router_decision == "INCOMPLETE":
            console.print(f"\n=== STEP 2: VALIDATION ===\nMissing Information:{missing_information}")
            log_event(f"\n=== STEP 2: VALIDATION ===\nMissing Information:{missing_information}")

        return {"router_decision": router_decision, "retrieved_context": retrieved_context, "useful_information": useful_information, "missing_information": missing_information, "reasoning": reasoning, "loop_count": loop_count+1}

    def answer(self, state: GraphState):
        # print("\n=== STEP 3: ANSWERING ===")
        question = state["question"]
        context = state["retrieved_context"]

        # answer_chain = Prompts.ANSWER_QUESTION | r1
        # llm_output = answer_chain.invoke({"retrieved_context": context, "question": question}).content
        prompt = Prompts.ANSWER_QUESTION.invoke({"retrieved_context": context, "question": question}).text
        messages = [
            {"role": "user", "content": prompt}
        ]
        llm_output = r1.chat.completions.create(
            model="ep-20250208165153-wn9ft", messages=messages
        )
        reasoning = llm_output.choices[0].message.reasoning_content.strip()
        answer = llm_output.choices[0].message.content.strip()
        console.print(f"\n=== STEP 3: ANSWERING ===\nfinal_answer: {answer}")
        return {"answer_to_question": answer, "retrieved_context": context, "useful_information": state['useful_information'] }

    def find_missing_information(self, state: GraphState):
        # print("\n=== STEP 2b: FINDING MISSING INFORMATION ===")
        missing_information = state["missing_information"]
        # print("Searching for:", missing_information)
                
        tavily_query = self.tavily_client.search(missing_information, max_results=3)
        previously_retrieved_useful_information = state["useful_information"]
        newly_retrieved_context = "\n".join([r["content"] for r in tavily_query])

        console.print(f"\n=== STEP 2b: FINDING MISSING INFORMATION ===\nSearching for missing:{missing_information}\nNewly retrieved context: \n{newly_retrieved_context}")

        combined_context = f"{previously_retrieved_useful_information}\n{newly_retrieved_context}"
        # print("newly retrieved context:", newly_retrieved_context)
        return {"retrieved_context": combined_context}

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

    def gen_outline(self, question: str):
        # print("\n=== OUTLINES GENERATION ===")
        prompt = Prompts.OUTLINES_GEN.invoke({"question": question}).text
        messages = [
            {"role": "user", "content": prompt}
        ]
        llm_output = r1.chat.completions.create(
            model="ep-20250208165153-wn9ft", messages=messages
        )
        reasoning = llm_output.choices[0].message.reasoning_content.strip()
        answer = llm_output.choices[0].message.content.strip()

        console.print(f"\nInit Outlines:\n{answer}")
        log_event(f"\nInit Outlines:\n{answer}")
        return answer
    
    def get_final_report(self, outlines: list, results: list):
        assert len(outlines) == len(results)
        final_report = ''
        for outline, result in zip(outlines, results):
            final_report += f"# {outline}\n{result['answer_to_question']}\n\n"

        return final_report


    def run(self, question: str):
        # 先生成一个大纲，然后每个大纲去调用 workflow.invoke
        start_time = datetime.now()
        initial_logger(logging_path="logs", enable_stdout=False, log_file_name=f"{question}_{start_time.strftime('%Y%m%d%H%M%S')}")

        outlines = self.gen_outline(question)
        outlines_lst = re.findall(r'\d\.?\s*(.*?)(?=\n\d+\.|$)', outlines, re.DOTALL)
        outlines_lst = [item.strip() for item in outlines_lst][:1]

        results = parallel_process(
            outlines_lst,
            lambda x: self.workflow.invoke({"question": f"{question}-{x}"}, {"recursion_limit": 999}),
        )
        # result = self.workflow.invoke({"question": question})

        # 根据result的内容去拼接正文
        final_repot = self.get_final_report(outlines_lst, results)
        print("==== FINAL REPORT ====")
        print(final_repot)

        end_time = datetime.now()
        print(f"Total time: {end_time - start_time}")
        # Save report
        os.makedirs("output", exist_ok=True)
        with open(f"output/{question}_{start_time.strftime('%Y%m%d%H%M%S')}.md", "w") as f:
            f.write(final_repot)

        return final_repot

if __name__ == "__main__":
    agent = QAAgent()
    agent.run("Who is George Washington?")