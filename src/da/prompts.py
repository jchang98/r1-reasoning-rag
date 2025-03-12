
THOUGHT = """\
你是一个有脑子的 **SearchAgent**，根据交互历史，为下一步“[CALL]”做一次前置检索。
\n\n\n
### 交互历史：
{history}
\n\n\n
### 检索问句："""


PLAN = """\
根据用户问句，帮我制定一个 1-5 步 latex-algorithm 格式的信息收集的路径链。
\n\n\n
### 工具：
{tools}
\n\n\n
### 要求：
-> 不能出现代码；
-> 不能出现具体的 “function call”
-> 不需要总结模块；
-> 考虑已有的数据，是否需要新增数据；
-> 计划的详细程度应当与问题的复杂程度成正比；
    - 简单的取数 1 步（例如：某只股票的收盘价）
    - 复杂取数 3 步（例如：某公司的财务指标 或 大盘涨停的公司都属于那些行业）
    - 复杂分析最多 5 步（例如：某股票技术面分析）
-> 问句与工具无关，则拒绝规划；
\n\n\n
### 示例：
[USER]: 同花顺技术面指标
[PLAN]:
\\begin{{algorithm}}
\caption{{同花顺技术面分析信息收集路径链}}
\\begin{{algorithmic}}[1]
\STATE 获取同花顺股票数据
\IF{{数据源不包含股票技术面数据}}
    \STATE 使用 iFinD 工具提取同花顺的股票技术面数据（包括收盘价、K线图、均线等）
\ENDIF
\STATE 对提取的数据进行清理，筛选出技术面分析所需的关键指标（例如：RSI、MACD等）
\IF{{RSI 或 MACD 数据不存在}}
    \STATE 使用已有的收盘价等指标计算 RSI 和 MACD
\ENDIF
\STATE 分析数据趋势，计算技术指标，识别买卖信号
\IF{{数据趋势显现出强烈的买入或卖出信号}}
    \STATE 输出技术分析结果，结合市场环境给出投资建议
\ELSE
    \STATE 输出中性信号，提示观望
\ENDIF
\end{{algorithmic}}
\end{{algorithm}}

[USER]:帮我筛选几只预期年化2%，产品规模小于50亿，成立超过三年，申赎无限制的基金
[PLAN]:
\\begin{{algorithm}}
\caption{{筛选基金信息收集路径链}}
\\begin{{algorithmic}}[1]
\STATE 使用 iFinD 工具找到包含预期年化收益、产品规模、成立时间及申赎限制等信息的表
\STATE 对提取的基金数据进行清理。
\STATE 在筛选出符合条件的基金
\end{{algorithmic}}
\end{{algorithm}}
\n\n\n
### 交互历史：
{history}
\n\n\n
**[PLAN]**:"""


EXEC_SQL = """\
「SQL执行器」
我们将所有pandas.DataFrame类型的变量建成了数据库，你只需要提供一条sql，格式如下：
```sql
<your sql>
```
-> 输出：sql的执行结果。"""


EXEC_LAMBDA = """\
「Lambda表达式执行器」
> pip install pandas numpy scipy statsmodels ta-lib plotly
提供一个call，格式如下：
```json
{
    "description": 「describe what you are going to do in this step」,
    "packages": 「import necessary packages here」,
    "expression": 「your lambda expression, lambda 表达式不一定需要输入，若有输入，则需要在`input`中明确入参名和变量名」,
    "input": {
        input_name: var_name
        ...
    }
}
```
-> examples:
```json
{
    "description": "Perform fuzzy search within a DataFrame column using a lambda expression with regex support.",
    "packages": "import pandas as pd, numpy as np, re",
    "expression": "lambda df,column_name,search_term: df[df['column_name'].apply(lambda x: bool(re.search(search_term, x)))]",
    "input": {
        "df": "data_frame",
        "column_name": "target_column",
        "search_term": "regex_pattern"
    }
}
```
-> 输出：lambda表达式的执行结果。"""


IFIND_QUERY = """\
「iFinD经济、金融取数工具」
下一步动作可以是对取数工具 `ifind_data` 的“[CALL]”，格式如下：
```json
{
    "call": {
        "function": "ifind_data",
        "params": {
            "query": 「自然语言取数问句」
        }
    }
}
```
取数问句有如下几种类型：
    a. 带有标的、时间区间的问句（例如：“同花顺 [start date] - [end date](默认3年) 收盘价”）
    b. 带筛选条件问句（例如：“今日A股涨停的股票”）
    c. 宏观经济库，支持语义拆解（例如：“影响黄金价格的因素”）
请尝试根据你的理解改写或分解“[USER]”的问句（例如：`财务指标`->`现金流量`+`负债`+`利润` or `走势` -> `每日行情数据`）。"""


EXEC_CODEBLOCK = """\
【Python代码执行器】
conda instal pandas numpy plotly
conda install scipy statsmodels
你只需要提供一个代码块，格式如下：
```python\n<代码（使用 print 查看中间变量）>\n```
-> 输出：代码块中被 print 的变量。"""


ACTION = """\
你正处于一个 **ReAction** 流中，参考 PLAN 给出的行动链，根据交互历史给出下一步动作，通过取数、筛选、计算，收集信息。
\n\n\n\n
### 工具：
{tools}
\n\n\n\n
### 要求：
-> 当前时间：{now}
-> 清洗数据；
-> 由于数据源并非绝对可靠，我强制你反思上一步[CALL]的结果；
-> 根据用户问题“[USER]”和交互记录，给出下一步动作；
-> 在你提供动作“[CALL]”之后,我会给你提供动作执行的结果“[GOT]”；
-> 若你认为信息充足，则使用以下代码退出整个 ReAction 流：
```python
exit()
```
-> 在 Reaction 流中，不要用`.show()`或`.render()`等方法渲染 Figure，这可能会导致性能下降；
\n\n\n\n
### 交互历史：
{history}
\n\n\n\n
**[CALL]**:"""


# -> 下一步动作还可以是使用搜索引擎 `search`，格式如下：
# ```json
# {{
#     "call": {{
#         "function": "search",
#         "params": {{
#             "query": <any search query>
#         }}
#     }}
# }}
# -> 下一步动作也可以是对SQL执行器的“[CALL]”，格式如下：
# ```json
# {{
#     "description": <describe what you are going to do in this step>,
#     "call": {{
#         "function": "exe_sql",
#         "params": {{
#             "query": <your sql query, no wrapping!>
#         }}
#     }}
# }}
# ```
# \n\n\n
# ### SQL 编译器如下：
# def exe_sql(query):
#     conn = sqlite3.connect(":memory:")
#     cursor = conn.cursor()
#     dataframes = {{name: obj for name, obj in globals().items() if isinstance(obj, pd.DataFrame)}}
#     for var_name, df in dataframes.items():
#         table_name = var_name
#         df.to_sql(table_name, conn, index=False, if_exists='replace')
#     result = pd.read_sql_query(query, conn)
#     conn.close()
#     return result


# -> 参考资料：
#     "lambda: go.Figure(data=go.Heatmap(z=df3.select_dtypes(include=[np.number]).corr().values, x=df3.select_dtypes(include=[np.number]).columns, y=df3.select_dtypes(include=[np.number]).columns, colorscale='Viridis')).update_layout(title='Correlation Heatmap of Financial Metrics', xaxis_title='Metrics', yaxis_title='Metrics')"
#     "lambda: go.Figure().add_trace(go.Scatter(x=df3['Date'], y=result_37['RSI'], mode='lines', name='RSI')).add_trace(go.Scatter(x=df3['Date'], y=result_37['MACD'][0], mode='lines', name='MACD Line')).add_trace(go.Scatter(x=df3['Date'], y=result_37['MACD'][1], mode='lines', name='Signal Line')).add_trace(go.Bar(x=df3['Date'], y=result_37['MACD'][2], name='MACD Histogram')).update_layout(title='... RSI and MACD', xaxis_title='Date', yaxis_title='Value')"
# \n\n\n
# -> 只使用 plotly 作图，一次 [CALL] 只生成一张图；
# -> 请不要在 **ReAction** 流中使用 `.show()` / `.render()` 等方法展示图表，这会导致进程中断； 
# -> 如果数据中有空值“--”，先清洗数据；
# -> 绘图时如果图中的时间序列不在一个量级，会导致量级小的指标呈现效果很差。根据绘图的结果，决定是否使用双 yaxis 轴，以处理不同量级的时间序列；
# -> 一个图最多两个 yaxis 轴，因此三个量级各异的时间序列不要画在一个图里；
# -> 多使用二次计算指标（而非表中原本存在的列）；
# ACTION = """\
# 你正处于一个 **ReAction** 流中，根据交互历史给出下一步动作（注意：只前进一步），为用户搜集所需的数据。
# \n\n\n
# ### 我为你安装了这些外部库：
# conda instal pandas
# conda install scipy statsmodels
# \n\n\n
# ### Lambda 表达式编译器如下：
# def exe_lambda(lambda_expression):
#     return eval(lambda_expression)()
# \n\n\n
# ### 要求：
# -> 根据用户问题“[USER]”和交互记录，给出下一步动作；
# -> 下一步动作可以是对取数工具 `get_data` 的“[CALL]”，格式如下：
# ```json
# {{
#     "call": {{
#         "function": "get_data",
#         "params": {{
#             "query": <自然语言取数问句>
#         }}
#     }}
# }}
# ```
# 取数问句有如下几种类型：
#     a. 带有标的、时间区间的问句（例如：“同花顺 近一周 收盘价”）
#     b. 带筛选条件问句（例如：“今日A股涨停的股票”）
#     c. 宏观经济库，支持语义拆解（例如：“影响黄金价格的因素”）
# 请尝试根据你的理解改写或分解“[USER]”的问句（例如：`财务指标`->`现金流量`+`资产负债`+`利润`）

# -> 下一步动作也可以是对lambda表达式执行器的“[CALL]”，格式如下：
# ```json
# {{
#     "description": <describe what you are going to do in this step>,
#     "call": {{
#         "function": "exe_lambda",
#         "packages": <import xxx as xxx; from xxx import xxx ...>,
#         "params": {{
#             "lambda_expression": < 此处格式为 —— `lambda: ...`（注意 lambda 表达式不是代码块，控制在一行内解决）>
#         }}
#     }}
# }}
# ```

# -> ReAction 流中的每个 lambda 表达式都应当返回非 `None` 变量（尤其绘图时）；
# -> lambda 表达式尽量不要太长，复杂变量你可以分多步生成；
# -> 在你提供动作之后,我会给你提供动作执行的结果“[GOT]”；
# -> 若你认为信息充足，则使用 ```python\\nlambda: exit()\\n``` 退出整个 ReAction 流；
# \n\n\n
# ### 交互历史：
# {history}
# \n\n\n
# **[CALL]**:"""


# \n\n\n
# ### 要求：
# -> 变量引用格式为 —— 尖括号内含变量名，以下是一个示例：
# \"\"\"
# ba la ba la ba la ~ ~ ~
# 下表展示了...
# <result_\d+>
# ba la ba la ba la ~ ~ ~
# \"\"\"；
# -> r'<result_\d+>' 是变量名的占位符；
ANSWER = """\
通过一系列工具调用，我们获得了一些结果。现在请你根据这些数据简要回答用户问题。
\n\n\n
### 要求：
-> 在最后引用能够支撑结论的"DataFrame"（用clean和join后的表），格式为：
\"\"\"\n---\n**「source」**\n\nresult_\d+\n\nresult_\d+...\"\"\"
\n\n\n
### 交互历史：
{history}
\n\n\n
### 回答："""