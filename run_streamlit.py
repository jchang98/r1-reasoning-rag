import streamlit as st
import subprocess
import asyncio
from datetime import datetime
from src.agent import QAAgent

st.set_page_config(layout="wide")
st.title('📖 Hithink Deep Research V3')

agent = QAAgent()

def clean():
    # 保持用户输入为空
    if "user_input" in st.session_state:
        st.session_state['user_input'] = []
    if "start_time" in st.session_state:
        st.session_state["start_time"] = ""
    if "message_queue" in st.session_state:
        st.session_state['message_queue'] = []
    
    # 使用rerun来刷新整个页面
    st.rerun()



    
def main():
    if 'user_input' not in st.session_state:
        st.session_state['user_input'] = []
    if "start_time" not in st.session_state:
        st.session_state["start_time"] = ""
    if "message_queue" not in st.session_state:
        st.session_state['message_queue'] = []

    model = st.sidebar.selectbox('选择模型', ['deepseek-r1'])
    is_polish = st.sidebar.selectbox('是否润色', [False, True])
    polish_step = st.sidebar.slider('润色步数', min_value=2, max_value=8, value=4)
    max_outline_num = st.sidebar.slider('大纲数', min_value=1, max_value=8, value=8)
    max_loop = st.sidebar.slider('最大循环数', min_value=1, max_value=10, value=5)
    clear = st.sidebar.button("clear")
    if clear:
        clean()

    user_input = st.chat_input("Enter a question:")


    if user_input :

        with st.chat_message("user"):
            st.markdown(user_input)
        start_time = datetime.now()
        agent.run(user_input, polish=is_polish, polish_step=polish_step, max_outline_num=max_outline_num, max_loop=max_loop)

        st.session_state['user_input'].append(user_input)
        st.session_state['start_time'] = start_time
                  


if __name__ == '__main__':
    main()
