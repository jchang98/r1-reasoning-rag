from src.da.basellm import *
from src.da.prompts import *

import scipy.stats as stats
import stattools
import statsmodels
import arch

import numpy as np
import pandas_datareader as pdr
import yfinance as yf
import talib
import QuantLib as ql
import empyrical as emp
import pyfolio as pf
from zipline.api import order, record, symbol

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

import mplfinance as mpf
import finplot as fplt

from pyecharts.charts import (
    Line,
    Bar,
    Pie,
    Scatter,
    Kline,
    Radar,
    Funnel,
    Gauge,
    Tree,
    Graph,
    Sankey,
    Boxplot,
    Candlestick,
    EffectScatter,
    Parallel,
    Sankey,
    Sunburst,
    Liquid,
    ThemeRiver,
    WordCloud,
)
from pyecharts import options as opts
from pyecharts.globals import ThemeType, SymbolType
from pyecharts.commons.utils import JsCode
from pyecharts.components import Table

from streamlit.components.v1 import html
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import pandas as pd
pd.set_option('display.max_rows', 30)

import json
import sys
import io
import random
import argparse
import traceback
import sqlite3
import re


def exe_sql(query):
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    dataframes = {name: obj for name, obj in globals().items() if isinstance(obj, pd.DataFrame)}
    for var_name, df in dataframes.items():
        table_name = var_name
        df.to_sql(table_name, conn, index=False, if_exists='replace')
    result = pd.read_sql_query(query, conn)
    conn.close()
    return result


def clean_dataframe(df: pd.DataFrame):
    ### TODO: 1、将列名里的`%`去掉（如果存在）；
    ### TODO: 2、类型为 str 的单元格，去除前后空格；
    df.replace("--", np.nan, inplace=True)
    
    percent_pattern = re.compile(r'^[+-]?\d+(\.\d+)?%$')
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in object_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].apply(lambda x: float(x.rstrip('%')) / 100 if percent_pattern.match(x) else x)
    
    number_with_commas_pattern = re.compile(r'^[+-]?\d{1,3}(,\d{3})*(\.\d+)?$')
    for col in object_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].apply(lambda x: float(x.replace(',', '')) if number_with_commas_pattern.match(x) else x)
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='ignore')
    
    if df.shape[0] > 0:
        last_row = df.iloc[-1].astype(str)
        if last_row.str.contains('数据来源').any():
            df = df.iloc[:-1].reset_index(drop=True)
    
    date_patterns = [
        re.compile(r'^\d{4}-\d{1,2}-\d{1,2}$'),  # YYYY-MM-DD
        re.compile(r'^\d{1,2}/\d{1,2}/\d{4}$'),  # DD/MM/YYYY 或 MM/DD/YYYY
        re.compile(r'^\d{1,2}-\d{1,2}-\d{4}$'),  # DD-MM-YYYY 或 MM-DD-YYYY
    ]
    
    date_cols = []
    for col in df.columns:
        date_like = df[col].dropna().apply(
            lambda x: any(pattern.match(str(x).strip()) for pattern in date_patterns)
        )
        if date_like.mean() > 0.9:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                invalid_ratio = df[col].isna().sum() / df[col].notna().sum()
                if invalid_ratio > 0.1:
                    date_cols.append(col)
            except Exception as e:
                date_cols.append(col)
    
    if date_cols:
        df.drop(columns=date_cols, inplace=True)
    
    cols_to_drop = []
    for col in df.columns:
        non_null = df[col].dropna()
        type_counts = non_null.map(type).value_counts()
        if len(type_counts) > 1:
            if type_counts.max() / type_counts.sum() < 0.9:
                cols_to_drop.append(col)
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
    
    all_date_cols = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()
    if all_date_cols:
        df.dropna(subset=all_date_cols, inplace=True)
    
    df.dropna(how='all', axis=1, inplace=True) 
    if len(df.columns) > 1:
        df.dropna(subset=df.columns[1:], how='all', inplace=True)    
    if len(df) > 1:
        cols_to_drop = [col for col in df.columns if df[col][1:].isna().all()]
        df.drop(columns=cols_to_drop, inplace=True)

    df = df.drop(columns=df.filter(regex="Unnamed").columns.tolist())
    
    return df


class DataAgent():
    
    def __init__(
            self, 
            acsv_files, 
            client, 
            history=None, 
            results=None,
            answer=None,
            max_call_turns=5,
            max_debug_turns=3,
        ):
        self.client = client
        self.results = results if results is not None else {}
        self.dfs = {}
        self.max_call_turns = max_call_turns
        self.max_debug_turns = max_debug_turns

        # FILES
        for idx, acsv in enumerate(acsv_files):

            key = f"df{idx}"
            if acsv.name.endswith(".csv"):
                df = pd.read_csv(acsv)
            elif acsv.name.endswith(".xlsx"):
                df = pd.read_excel(acsv, engine="openpyxl")    
            elif acsv.name.endswith(".xls"):
                df = pd.read_excel(acsv, engine="xlrd")   
            else:
                raise ValueError(f"unsupported file type `{acsv.name}`")          
            
            self.dfs[key] = df  # clean_dataframe(df)
            self.results[key] = self.dfs[key]
        
        # GIVEN
        givens = []
        for idx, (key, df) in enumerate(self.dfs.items()):
            table_info = f"""\
**[GIVEN]**: <{acsv_files[idx].name.split('/')[-1]}>
Var: <{key}>; Type: <class 'pandas.DataFrame'>;
<{key}.head(5).to_csv()>:
```csv
{df.head(5).to_csv()}```;
Dtypes:
```csv
{df.dtypes.to_csv()}```;\n\n"""
            globals().update({key: df})
            givens.append(table_info)
        self.given_length = len(givens)
        self.history = history if history else givens
        self.truncate_history()
        self.answer = answer if answer else ""

    def truncate_history(self, max_length=6):
        # TODO: replace variables in answer
        givens = self.history[:self.given_length]
        after_givens = self.history[self.given_length:]
        conversation_history = []; 
        idx=0

        while idx < len(after_givens):
            h = after_givens[idx]
            if h.startswith("**[USER]**"):
                conversation_history.append(h)
            elif h.startswith("**[ANSWER]**"):
                var_pattern = re.compile(r'<(result_\d+)(?:\[(\d+)\])?>')
                matches = var_pattern.finditer(h)

                for match in matches:
                    var_name = match.group(0)
                    conversation_history.extend([g for g in after_givens if var_name in g and g.startswith("**[GOT]**")])
                
                conversation_history.append(h)
            idx += 1

        if len(conversation_history) > max_length:
            conversation_history = conversation_history[-max_length:]     
        self.history = givens + conversation_history

    def call(self, question=None):
        self.history.append(f"**[USER]**:\n{question}\n\n")

        _, callstarts = max(((k, int(m.group(1)) if (m := re.match(r'result_(\d+)', k)) else 0) for k in self.results), key=lambda x: x[1])
        callstarts += 5
        
        i = callstarts
        while i < callstarts + self.max_call_turns:
            failed = "Failed" in self.history[-1]
            prompt = ACTION.format(
                history="\n".join(self.history)
            )

            response = self.client.ask([{"role": "user", "content": prompt}])
            self.history.append(f"{response}\n")

            try:
                if "exit()" in response:
                    self.history = self.history[:-1]
                    break
                else:
                    if "json" in response:
                        action = extract(response, "json")
                        query = action.get("params", None); 
                        assert query is not None, "where is your sql?"
                        var_name = action.get("return", None); 
                        assert var_name is not None, "please provide name for the execution return"
                        globals()[var_name] = exe_sql(query=query)
                        result = globals().get(var_name, None)

                        self.history.append(f"**[GOT]**:\n{result}")
                    else:
                        action = extract(response, "python")
                        captured_print = io.StringIO()
                        sys.stdout = captured_print
                        exec(action, globals())
                        sys.stdout = sys.__stdout__

                        self.history.append(f"**[GOT]**:\n{captured_print.getvalue()}")
                    
                if failed:
                    roll_back_idx = next((i for i, h in enumerate(self.history) if "Failed" in h), None) - 1
                    del self.history[roll_back_idx: -2]

            except Exception as e:
                self.history.append(f"**[GOT]**:\n< Failed: \"{str(e)[:2048]}\" >\n{traceback.format_exc(5)}\n\n")
            
            i += 1
        
        prompt = ANSWER.format(history="".join(self.history))
        self.answer = self.client.ask([{"role":"user", "content":prompt}])
        self.history.append(f"**[ANSWER]**\n{self.answer}\n\n\n\n")


if __name__ == "__main__":
    # df = pd.read_csv("../浦发银行(600000.SH)_每日行情数据统计.csv")
    # df = pd.read_excel("../浦发银行(600000.SH)-估值分析明细.xlsx")
    df = pd.read_excel("../../上证50-历史价格.xls")
    df = clean_dataframe(df)
    print(df.dtypes)
    print(df)