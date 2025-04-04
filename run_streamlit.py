import streamlit as st
import subprocess
import asyncio
from datetime import datetime
from src.agent import QAAgent, get_feedback
from src.utils import show_hist_log, parse_file_names, remove_if_exist
import os
import glob

st.set_page_config(layout="wide")
st.title('📖 Hithink Deep Research V3')
WORKING_DIR = "./local_mem"

def clean():
    # 保持用户输入为空
    if "user_input" in st.session_state:
        st.session_state['user_input'] = []
    if 'input_type' in st.session_state:
        st.session_state['input_type'] = ""
    if "start_time" in st.session_state:
        st.session_state["start_time"] = ""
    if "feedback" in st.session_state:
        st.session_state['feedback'] = ""
    if "message_queue" in st.session_state:
        st.session_state['message_queue'] = []


    if 'selected_problem' in st.session_state:
        st.session_state.selected_problem = None
    if 'selected_time' in st.session_state:
        st.session_state.selected_time = None


    remove_if_exist(f"{WORKING_DIR}/vdb_entities.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_full_docs.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_llm_response_cache.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_text_chunks.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_community_reports.json")
    remove_if_exist(f"{WORKING_DIR}/graph_chunk_entity_relation.graphml")
    
    # 使用rerun来刷新整个页面
    st.rerun()



    
def main():
    if 'user_input' not in st.session_state:
        st.session_state['user_input'] = []
    if 'input_type' not in st.session_state:
        st.session_state['input_type'] = ""
    if "start_time" not in st.session_state:
        st.session_state["start_time"] = ""
    if "feedback" not in st.session_state:
        st.session_state['feedback'] = ""
    if "message_queue" not in st.session_state:
        st.session_state['message_queue'] = []

    remove_if_exist(f"{WORKING_DIR}/vdb_entities.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_full_docs.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_llm_response_cache.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_text_chunks.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_community_reports.json")
    remove_if_exist(f"{WORKING_DIR}/graph_chunk_entity_relation.graphml")

    model = st.sidebar.selectbox('选择模型（除写作外）', ['deepseek-r1-vol', 'deepseek-v3-vol', 'deepseek-r1-ths'])
    writing_model = st.sidebar.selectbox('选择写作模型', ['deepseek-v3-vol', 'deepseek-r1-vol', 'deepseek-r1-ths'])
    data_eng = st.sidebar.selectbox('数据引擎', ['none', 'ifind_data', 'ifind_data_agent'])
    # writing_method = st.sidebar.selectbox('生成方式', ["parallel", "polish", "serial"]) # 并行生成，润色，串行生成
    writing_method = st.sidebar.multiselect('生成方式', ["parallel", "polish", "serial"]) # 并行生成，润色，串行生成
    if writing_method == "polish":
        polish_step = st.sidebar.slider('润色步数', min_value=2, max_value=8, value=4)
    else:
        polish_step = -1
    max_outline_num = st.sidebar.slider('大纲数', min_value=1, max_value=8, value=8)
    max_loop = st.sidebar.slider('最大循环数', min_value=1, max_value=10, value=5)
    clear = st.sidebar.button("clear")
    if clear:
        user_input =  ""
        clean()


    #查找历史
    files = glob.glob(os.path.join(f"logs", '*')) 
    problems, times = parse_file_names(files)

    if 'selected_problem' not in st.session_state:
        st.session_state.selected_problem = None
    if 'selected_time' not in st.session_state:
        st.session_state.selected_time = None

    st.sidebar.selectbox(
        "问题时间",
        options=[""] + list(times.keys()),  
        key="selected_time",
        on_change=lambda: st.session_state.update(selected_promblem=None)  
    )

    if st.session_state.selected_time:
        st.sidebar.selectbox(
            "问题记录",
            options=[""] + times[st.session_state.selected_time],  # 添加空字符串作为默认选项
            key="selected_promblem"
    )
    show_history = st.sidebar.button("查询历史")
    if show_history:
        #print(st.session_state.selected_problem+st.session_state.selected_time)
        show_hist_log(show_file=f"{st.session_state.selected_promblem}")




    user_input = st.chat_input("Enter a question:")

    if user_input and st.session_state['input_type'] == "":
        # 第一次输入，需要澄清用户的意图
        start_time = datetime.now()
        feedback = asyncio.run(get_feedback(question=user_input, model=model, start_time=start_time))


        st.session_state['input_type'] = "feedback"
        st.session_state['user_input'].append(user_input)
        st.session_state['feedback'] = feedback
        st.session_state['start_time'] = start_time

    elif user_input and st.session_state['input_type'] == "feedback":
        # 根据feedback正式开始生成
        user_input_orig = st.session_state['user_input'][-1]
        feedback = st.session_state['feedback']
        feedback_answer = user_input
        start_time = st.session_state['start_time']


        agent = QAAgent(writing_method, model, writing_model, data_eng)
        with st.chat_message("user"):
            st.markdown(user_input)
        asyncio.run(agent.run(user_input_orig, writing_method=writing_method, polish_step=polish_step, max_outline_num=max_outline_num, max_loop=max_loop, feedback=feedback, feedback_answer=feedback_answer, start_time=start_time))

        st.session_state['input_type'] = ""
                  


if __name__ == '__main__':
    main()
