import re
import streamlit as st
import uuid
from streamlit.components.v1 import html

import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

from pyecharts.charts import (
    Line, Bar, Pie, Scatter, Kline, Radar, Funnel, Gauge, Tree,
    Graph, Sankey, Boxplot, Candlestick, EffectScatter, Parallel, Sunburst, Liquid,
    ThemeRiver, WordCloud
)


def extract_result_key(got_entry):
    match = re.search(r'Var: <(result_\d+)>', got_entry)
    if match:
        return match.group(1)
    else:
        return None    


def extract(content:str, pattern="json"):
    content = content.replace("\n\n", "\n")
    if pattern is not None:
        regex_pattern = r'```' + pattern + r'\s+(.*?)```'
        blocks = re.findall(regex_pattern, content, re.DOTALL)
        if not blocks:
            blocks = [content]
        block = blocks[0].strip()
        if pattern == "json":
            try:
                json_variable = json.loads(block)
                return json_variable
            except json.JSONDecodeError as e1:
                try: 
                    json_variable = eval(block)
                    return json_variable
                except Exception as e2:
                    raise ValueError("Cannot parse your toolcall as JSON") from e2
        else:
            return block
    else:
        return content


def render(obj):
    try:
        unique_key = str(uuid.uuid4())

        if isinstance(obj, plt.Figure):
            fig = obj
        elif isinstance(obj, plt.Axes):
            fig = obj.figure
        elif isinstance(obj, plt.Artist):
            if obj.axes:
                fig = obj.axes.figure
        elif isinstance(obj, go.Figure):
            st.plotly_chart(obj, use_container_width=True, key=unique_key)
        elif hasattr(obj, 'to_plotly_json'):
            fig = go.Figure(obj)
            st.plotly_chart(fig, use_container_width=True, key=unique_key)
        elif isinstance(obj, pd.DataFrame):
            st.dataframe(obj)
        elif hasattr(obj, 'render_embed') and callable(getattr(obj, 'render_embed')):
            chart_html = obj.render_embed()
            html(chart_html, height=600, scrolling=True)
        elif hasattr(obj, 'to_html') and callable(getattr(obj, 'to_html')):
            fig_html = obj.to_html()
            html(fig_html, height=600, scrolling=True)
        elif hasattr(obj, 'to_dict') and 'mark' in obj.to_dict():
            st.altair_chart(obj)
        elif hasattr(obj, 'show') and callable(getattr(obj, 'show')):
            obj.show()
            st.pyplot()
        else:
            st.write(obj)
    
    except Exception as e:
        st.error(f"无法渲染对象: {e}")


def parse_history_into_rounds(history):
    """
    Each round contains [USER, CALL-GOT-Loop and ANSWER]。
    """
    rounds = []
    current_round = {}
    for entry in history:
        if entry.startswith("**[USER]**"):
            if current_round:
                rounds.append(current_round)
                current_round = {}
            current_round['user'] = entry
        elif entry.startswith("**[CALL]**") or entry.startswith("**[DEBUG]**") or entry.startswith("**[GOT]**"):
            if 'calls' not in current_round:
                current_round['calls'] = []
            current_round['calls'].append(entry)
        elif entry.startswith("**[ANSWER]**"):
            current_round['answer'] = entry
        elif entry.startswith("**[PLAN]**"):
            current_round['plan'] = entry
    if current_round:
        rounds.append(current_round)
    return rounds


def render_round(rnd, results):
    if 'user' in rnd:
        st.markdown(rnd['user'])
    
    # if 'plan' in rnd:
    #     st.markdown(rnd['plan'])

    if 'calls' in rnd:
        try:
            with st.expander("DETAILED CALLs"):
                for entry in rnd['calls']:
                    if entry.startswith("**[CALL]**") or entry.startswith("**[DEBUG]**"):
                        try:
                            call_content = extract(entry, "json")
                            description = call_content.get('description', '')
                            st.write(f"**[CALL]**")
                            st.json(call_content)
                        except Exception as e:
                            st.write(entry)
                    elif entry.startswith("**[GOT]**"):
                        if "Failed" in entry:
                            error_msg = re.search(r'< Failed: (.+?) >', entry)
                            if error_msg:
                                st.error(f"**[GOT]**\n{error_msg.group(1)}")
                            else:
                                st.error(entry)
                        else:
                            st.write("**[GOT]**")
                            result_key = extract_result_key(entry)
                            if result_key in rnd['answer']:
                                st.write("📈")
                            else:
                                if result_key and result_key in results:
                                    try:
                                        render(results[result_key])
                                    except Exception as e:
                                        st.write(str(results[result_key]))
                                else:
                                    st.write(entry)
        except Exception as e:
            st.error(f"Error in calls: {e}")

    if 'answer' in rnd:
        answer_text = rnd['answer'].replace("**[ANSWER]**\n", "")
        var_pattern = re.compile(r'(result_\d+)(?:\[(\d+)\])?')
        matches = var_pattern.finditer(answer_text)

        last_index = 0
        for match in matches:
            var_full = match.group(0)
            var_key = match.group(1)
            var_index = match.group(2)

            text_part = answer_text[last_index:match.start()]
            if text_part:
                st.markdown(f"{text_part}", unsafe_allow_html=True)

            if var_key in results:
                obj = results[var_key]
                if var_index is not None:
                    try:
                        obj = obj[int(var_index)]
                    except (IndexError, ValueError, TypeError) as e:
                        st.error(f"cannot acccess {var_key}[{var_index}] : {e}")
                        obj = None
                
                render(obj)
            else:
                st.markdown(var_full, unsafe_allow_html=True)

            last_index = match.end()

        remaining_text = answer_text[last_index:]
        if remaining_text:
            st.markdown(f"{remaining_text}", unsafe_allow_html=True)

