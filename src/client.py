from typing import Literal, Sequence, Optional, List, Union
import requests
import json
from datetime import datetime
import pandas as pd
pd.set_option('display.max_rows', 30)
import asyncio
import httpx

from src.da.basellm import *
from src.da.prompts import *
from src.da.agent_plus import *

class TableWithDescription():
    def __init__(self, table: pd.DataFrame, description:str, content:str):
        self.table = table
        self.description = description
        self.content = content


class DataClient:
    """
    DataClient is a client for the data.
    """

    async def ifind_data(self, query: str, max_results: int = 4) -> list:
        await asyncio.sleep(1)
        url = "http://open-server.51ifind.com/standardgwapi/arsenal_service/ifind-python-aime-tools-service/get_data"
        headers = {
            "X-Arsenal-Auth":"arsenal-tools",
            "X-Switch": "enable_pick_result=0;enable_f9_data_agent_answer=0",
            'Cookie': "osgw_app_id=b894fa1f3b044c35bd6220ba1d434fa5;osgw_uid=L20;osgw_udid=L20",
            "x-ft-arsenal-auth": "L24FB1H14W54KQENSSPC4CSB2S0PPM5M",
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        params = {"query":query}

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, data=params, headers=headers)
            sources = json.loads(
                response.json()['data']['query_data']['condition']
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
                        df_str = df.to_string()
                        content = f"【{description}】\n{df_str}"
                        # print(f"【{description}】")
                        # print(df.head())
                        table_with_description_list.append({
                                "table": df_str,
                                "description": description,
                                "content": content,
                                "type": "index"
                            })
                    except Exception as e:
                        print(f"ifind_data: parser error:{e}")
                        return []
            return table_with_description_list[:max_results] if len(table_with_description_list) else []
        except Exception as e:
            print(f"ifind_data: requests error:{e}")
            return []
    

    async def ifind_data_agent(self, query: str):
        try:
            client = OpenAIClient(verbose=False)
            agent = DataAgent(
                client=client, 
                tools=[EXEC_LAMBDA, IFIND_QUERY],
                acsv_files=list(), 
                history=list(), 
                results=dict(),
                answer=str(),
            )
            await agent.call(query)

            for k, v in agent.results.items():
                globals()[k] = v

            # 将agent.answer中的var和 Type 删除掉
            final_res = re.sub(r"\nVar: \<result\_\d*\>; Type: \<class .*\>;", "", agent.answer)
            final_res = re.sub(r"result\_\d*", "", final_res)
            return [{"content": final_res, "type": "index"}]
        except Exception as e:
            print(f"ifind_data_agent: requests error:{e}")
            return []

class SearchClient:
    """
    SearchClient is a client for the bing search.
    """

    def __init__(self, se: Optional[str] = "BING"):
        self.se = se
        self.url = 'https://tgenerator.aicubes.cn/iwc-index-search-engine/search_engine/v1/search'
        self.header = {
            'X-Arsenal-Auth': 'arsenal-tools'
        }
    
    async def search(self, query: str, max_results: int = 2, **kwargs):
        data = {
            "query": query,
            "se": self.se,
            "limit": max_results * 4,
            "user_id": "test",
            "app_id": "test",
            "trace_id": "test",
            "with_content": True
        }
        if kwargs:
            data.update(kwargs)
        try:
            async with httpx.AsyncClient() as client:
                response_dic = await client.post(self.url, data=data, headers=self.header)
            if response_dic.status_code == 200:
                response = json.loads(response_dic.text)['data']


                response = sorted(response, key=lambda x: x['publish_time'] if x['publish_time'] else float("-inf"),reverse=True)

                # 替换为serapi googlesearch的格式
                organic_results_lst = []
                for idx, t in enumerate(response):
                    position = idx +1
                    title = t['title'] if t['title'] else ""
                    link = t['url']
                    snippet = t['summary'] if t['summary'] else ""
                    date = t['publish_time'] if t['publish_time'] else ""
                    source = t['data_source'] if t['data_source'] else ""
                    content = t['content'] if t['content'] else ""


                    if date:
                        dt_object = datetime.fromtimestamp(date)
                        formatted_time = dt_object.strftime("%Y-%m-%d %H:%M:%S")
                        date = formatted_time
                        

                    organic_results_lst.append({
                        "position": position,
                        "title": title,
                        "url": link,
                        "snippet": snippet,
                        "date": date,
                        "source": source,
                        "content": content,
                        "type": "search"
                    })

                return organic_results_lst[:max_results]

            else:
                print(f"搜索失败，状态码：{response_dic.status_code}")
                return []
        except Exception as e:
            print(f"搜索请求发生错误：{str(e)}")
            return []  # 出现异常时也返回空列表  