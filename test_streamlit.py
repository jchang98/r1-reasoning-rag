import streamlit as st

# html_file = "/mnt/data/jinchang/smolagents/r1-reasoning-rag-v4/r1-reasoning-rag/local_mem/graph_chunk_entity_relation.html"
# with open(html_file, 'r', encoding='utf-8') as file:
#     html_content = file.read()
# with st.chat_message("assistant"):
#     st.markdown(f"==== MINDMAP ====\n")
#     st.components.v1.html(html_content,height=600) 


t_json = {
    "t":1,
    "nodes":"# sadjfoa \n **sfads8** \n - asdfasdf \n - asdfasdf",
}
with st.chat_message("assistant"):
    st.write(t_json)