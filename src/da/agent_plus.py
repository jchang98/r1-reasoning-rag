from src.da.basellm import *
from src.da.prompts import *

from dotenv import load_dotenv
load_dotenv()
import sqlite3
import numpy as np
import pandas as pd
import talib
import statsmodels
import scipy
pd.set_option('display.max_rows', 30)

import re
import io
import sys
import time
import json

from prettytable import PrettyTable
from typing import List
from datetime import datetime
import asyncio
import httpx



class DuplicateActionError(Exception):
    pass


class EmptyDataFrameError(Exception):
    pass


class NoneReturnError(Exception):
    pass


class TableWithDescription():
    def __init__(self, table: pd.DataFrame, description:str):
        self.table = table
        self.description = description


def exe_lambda(action:dict, context:dict):
    exec(action.get('packages', ""))
    context.update(locals())

    lambda_expression = action['expression']
    lambda_input = action.get('input', {})

    if "lambda" not in lambda_expression:
        lambda_expression = f"lambda {','.join(list(lambda_input.keys()))}: " + lambda_expression
    resolved_inputs = {}
    for input_name, var_name in lambda_input.items():
        if var_name not in context:
            raise KeyError(f"variable '{var_name}' not found in globals()")
        resolved_inputs[input_name] = eval(var_name, context)

    result = eval(lambda_expression, context)(**resolved_inputs)
    return result


async def ifind_data(query: str, max_results: int = 4) -> List[TableWithDescription]:
    await asyncio.sleep(1)
    url = "http://open-server.51ifind.com/standardgwapi/arsenal_service/ifind-python-aime-tools-service/get_data"
    headers = {
        "X-Arsenal-Auth":"arsenal-tools",
        "X-Switch": "enable_pick_result=0;enable_f9_data_agent_answer=0",
        'Cookie': os.getenv("L20_COOKIE"),
        "x-ft-arsenal-auth": "L24FB1H14W54KQENSSPC4CSB2S0PPM5M",
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    params = {"query":query}

    # response = rq.post(url, data=params, headers=headers)
    async with httpx.AsyncClient() as client:
        response = await client.post(url, data=params, headers=headers)
    try:
        sources = json.loads(
            s=response.json()['data']['query_data']['condition']
        )['datas']

        table_with_description_list = list()
        for source in sources:
            source_info = source['datas']
            for tb_info in source_info:
                try:
                    description = ""
                    if tb_info.get("title"):
                        description = tb_info.get("title")
                    elif tb_info.get("description"):
                        description = tb_info.get("description")
                    tb = tb_info.get("data")
                    
                    if isinstance(tb, list):
                        tb = tb[0]
                    columns = tb['columns']
                    data = tb['data']

                    df = pd.DataFrame(data, columns=columns)
                    print(f"【{description}】")
                    print(df.head())
                    table_with_description_list.append(TableWithDescription(table=df, description=description))
                except Exception as e:
                    print(f"ifind_data: parser error:{e}")
                    return []
        return table_with_description_list[:max_results] if len(table_with_description_list) else None
    except Exception as e:
        print(f"ifind_data: parser error:{e}")
        return []


def search(query, context):
    url = 'https://tgenerator.aicubes.cn/iwc-index-search-engine/search_engine/v1/search'
    
    params = {
        'query': query,
        # 'se': 'BAIDU',
        'se': 'BING',
        'limit': 4,
        'user_id': 'test',
        'app_id': 'test',
        'trace_id': 'test',
        'with_content': True
    }

    header = {
        'X-Arsenal-Auth': 'arsenal-tools'
    }
    response_dic = rq.post(url, data=params, headers=header)

    if response_dic.status_code == 200:
        response =  json.loads(response_dic.text)['data']
        organic_results_lst = []
        for idx, t in enumerate(response):
            position = idx +1
            title = t['title'] if t['title'] else ""
            link = t['url'] 
            snippet = t['summary'] if t['summary'] else ""
            date = t['publish_time'] if t['publish_time'] else ""
            source = t['data_source'] if t['data_source'] else ""
            if date:
                dt_object = datetime.fromtimestamp(date)
                formatted_time = dt_object.strftime("%Y-%m-%d %H:%M:%S")
                date = formatted_time
            organic_results_lst.append({
                "position": position,
                "title": title,
                "link": link,
                "snippet": snippet,
                "date": date
            })
        res = {
            "search_parameters": response_dic.json()['header'],
            "organic_results": organic_results_lst
        }

        return res

    else:
        raise Exception(f"搜索失败，状态码：{response.status_code}")


def exe_sql(query, context):
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    dataframes = {name: obj for name, obj in context.items() if isinstance(obj, pd.DataFrame)}
    for var_name, df in dataframes.items():
        table_name = var_name
        df.to_sql(table_name, conn, index=False, if_exists='replace')
    result = pd.read_sql_query(query, conn)
    conn.close()
    return result


def prettytable(df: pd.DataFrame):
    table = PrettyTable()
    table.field_names = df.columns.tolist()
    for row in df.values:
        table.add_row(row)
    return table


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
            acsv_files: List[io.BufferedReader], 
            client, 
            tools, 
            history=None, 
            results=None,
            answer=None,
            max_call_turns=5,
            max_debug_turns=5,
        ):
        self.client = client
        self.tools = tools
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
{df}
Dtypes:
{df.dtypes.to_csv(index=False)};\n\n"""
            givens.append(table_info)
        self.given_length = len(givens)
        self.history = history if history else givens
        self.truncate_history()

        self.answer = answer if answer is not None else ""

    def truncate_history(self, max_length=10):
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
                var_pattern = re.compile(r'(result_\d+)(?:\[(\d+)\])?')
                matches = var_pattern.finditer(h)
                for match in matches:
                    var_name = match.group(0)
                    conversation_history.extend([g for g in after_givens if var_name in g and g.startswith("**[GOT]**") and g not in conversation_history])
                conversation_history.append(h)
            idx += 1

        if len(conversation_history) > max_length:
            conversation_history = conversation_history[-max_length:]     
        self.history = givens + conversation_history

    async def call(self, question=None):
        self.history.append(f"**[USER]**:\n{question}\n\n")

        # self.client = OpenAIClient(model_name="o1", verbose=True)
        # prompt = PLAN.format(
        #     tools="\n\n".join(self.tools), 
        #     history="\n".join(self.history)
        # )
        # response = self.client.ask([{"role": "user", "content": prompt}])
        # self.history.append(f"**[PLAN]**:\n{response}\n")

        if self.results:
            _, callstarts = max(((k, int(m.group(1)) if (m := re.match(r'result_(\d+)', k)) else 0) for k in self.results), key=lambda x: x[1])
        else:
            callstarts = 0
        callstarts += 5
        
        i = callstarts; j = 0; k = 0
        while i < callstarts + self.max_call_turns and j < self.max_debug_turns:
            
            if k == 2 and IFIND_QUERY in self.tools:
                self.tools.remove(IFIND_QUERY)

            failed = "Failed" in self.history[-1]
            # self.client = OpenAIClient(model_name="gpt-4o", verbose=True)
            now = time.ctime(time.time())
            prompt = ACTION.format(
                now=now,
                tools="\n\n".join(self.tools),
                history="\n".join(self.history)
            )
            response = await self.client.ask([{"role": "user", "content": prompt}])
            self.history.append(f"**[CALL]**:\n{response}\n")

            try:
                # if "exit()" in response: 
                #     break
                # else:
                #     action = extract(response, "json")

                #     if i > callstarts:
                #         if action == extract(next(h for h in reversed(self.history[:-1]) if "**[CALL]**" in h), "json"):
                #             raise DuplicateActionError("current toolcall has been performed already.")

                #     fct_call = action['call']

                #     fct = fct_call['function']
                #     params = fct_call['params']
                #     import_packages = fct_call.get('packages', None)

                #     if import_packages:
                #         exec(import_packages)

                #     context = {
                #         "exe_sql": exe_sql, 
                #         "exe_lambda": exe_lambda,
                #         "get_data": get_data,
                #     }
                #     context.update(locals())
                #     context.update(self.results)

                #     fct_ = context[fct]
                #     result = fct_(**params, context=context)
                if "exit" in response: 
                    break
                else:

                    if i > callstarts:
                        if response == next(h for h in reversed(self.history[:-1]) if "**[CALL]**" in h):
                            break
                        
                    context = {
                        "ifind_data": ifind_data,
                    }
                    context.update(globals())
                    context.update(self.results)
                    
                    if "```sql" in response:
                        action = extract(response, "sql")
                        result = exe_sql(action, context=context)

                    elif "```json" in response:
                        action = extract(response, "json")
                        if action.get("call"):
                            func = action['call']['function']
                            params = action['call']['params']
                            func_ = context[func]
                            result = await func_(**params)
                            k += 1
                        else:
                            result = exe_lambda(action, context=context)

                    elif "```python" in response:
                        action = extract(response, "python")
                        output = io.StringIO()
                        sys.stdout = output
                        exec(action, context); result = output.getvalue()
                        sys.stdout = sys.__stdout__

                    if result is None:
                        raise NoneReturnError("[CALL] 返回为空")

                    max_idx = 0
                    pattern = re.compile(r'^result_(\d+)$')
                    for key in self.results:
                        match = pattern.match(key)
                        if match:
                            current_max_idx = int(match.group(1))
                            if current_max_idx > max_idx:
                                max_idx = current_max_idx

                    if isinstance(result, list) and isinstance(result[0], TableWithDescription):
                        for j in range(len(result)):
                            result_key = f"result_{max_idx+1+j}"
                            tb_info = result[j]
                            self.results[result_key] = tb_info.table
                            # quoting=csv.QUOTE_ALL, 
                            got_entry = f"""\
**[GOT]**: {tb_info.description}
Var: <{result_key}>; Type: {type(tb_info.table)};
{tb_info.table}
Dtypes:
{tb_info.table.dtypes.to_csv(index=False)}\n\n"""
                            self.history.append(got_entry)
                    else:
                        result_key = f"result_{max_idx+1}"
                        self.results[result_key] = result
                        if isinstance(result, pd.DataFrame):
                            if result.empty:
                                raise EmptyDataFrameError("No data remain after call")
                            else:
                                # quoting=csv.QUOTE_ALL, 
                                h_text = f"**[GOT]**:\nVar: <{result_key}>; Type: {type(result)};\n{result}\n\n"
                            self.history.append(h_text)
                        else:
                            self.history.append(f"**[GOT]**:\nVar: <{result_key}>; Type: {type(result)}\n{str(result)[:2048]}\n\n")

                if failed:
                    roll_back_idx = next((i for i, h in enumerate(self.history) if "Failed" in h), None) - 1
                    i -= len(self.history[roll_back_idx: -2]); del self.history[roll_back_idx: -2]

            except Exception as e:
                self.history.append(f"**[GOT]**:\n< Failed: \"{str(e)}\" >\n\n") #\n{traceback.format_exc(20)}
                j += 1 
            i += 1
        
        # self.client = OpenAIClient(model_name="o1", verbose=True)
        prompt = ANSWER.format(history="".join(self.history))
        self.answer = await self.client.ask([{"role":"user", "content":prompt}])
        
        def replace_result(match):
            result_key = match.group(0)
            for entry in self.history:
                if result_key in entry:
                    return entry.replace("**[GOT]**:\n", "").split("Dtypes:")[0]
            return result_key

        self.answer = re.sub(r'result_\d+', replace_result, self.answer)
        self.history.append(f"**[ANSWER]**\n{self.answer}\n\n\n\n")


if __name__ == "__main__":
    result = ifind_data("东方财富技术指标分析")
    