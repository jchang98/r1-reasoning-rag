from typing import Literal, Sequence, Optional, List, Union
import requests
import json
from datetime import datetime

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
    
    def search(self, query: str, max_results: int = 5, **kwargs):
        data = {
            "query": query,
            "se": self.se,
            "limit": max_results,
            "user_id": "test",
            "app_id": "test",
            "trace_id": "test",
            "with_content": True
        }
        if kwargs:
            data.update(kwargs)
        try:
            response_dic = requests.post(self.url, data=data, headers=self.header)
            if response_dic.status_code == 200:
                response = json.loads(response_dic.text)['data']

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
                        "content": content
                    })

                return organic_results_lst

            else:
                print(f"搜索失败，状态码：{response_dic.status_code}")
                return []
        except Exception as e:
            print(f"搜索请求发生错误：{str(e)}")
            return []  # 出现异常时也返回空列表  