from openai import OpenAI
import os
import json
from blingfire import text_to_sentences_and_offsets
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import argparse
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


model_id = "deepseek-v3-250324"
# model_id = "deepseek-r1-250120"


from langchain_core.prompts import PromptTemplate
chunk = PromptTemplate(
        input_variables=["input"],
        template="""
        你是一个文本处理的帮手。我会输入一段文本，你需要从文本中提取出所有的关键主张或者主要信息点。

        # 要求
        - 理解句子的语义和上下文，确保每个输出句子都有意义和信息量。
        - 每个句子必须包括完整的主体、时间、事件等信息，对于markdown的表格和 mermaid语法的图片，请完整将图表切分到一个句子中。
        - 仅保留包含具体新闻、事件、数据等重要信息的句子，过滤掉没有信息量或观点的句子。
        - 请返回一个 json 列表。
        
        # 示例
        输入: **恒生科技指数**在2025年第一季度展现出强劲的市场表现，自1月13日的年内低点至3月13日累计涨幅达**34.90%**，显著超越同期**沪深300指数**（4.80%）和**中证1000指数**（16.63%）的涨幅。这一表现的核心驱动因素包括**南下资金**的持续流入（日均净买入43亿港元）以及**结构性行情**对科技股的聚焦——约20%的个股跑赢指数，主要集中在互联网和人工智能（AI）领域。不过，当前**ERP（股权风险溢价）**以美债利率衡量已回落至-1倍标准差区间，但结合中债利率的综合估值模型显示，港股尚未进入极端估值状态，仍存在**结构性机会**。从资金流向看，被动型ETF资金占据主导，而主动外资尚未大规模回流，这可能与**港元兑美元汇率**的短期波动有关。例如，2025年3月26日美元兑港元汇率报7.7763，过去三个月汇率波动幅度达0.3%，可能影响外资配置决策。\n\n```mermaid\npie\n    title 2027年中美台海军力对比预测\n    "中国海军水面舰艇" : 215\n    "美国印太舰队可部署舰艇" : 165\n    "解放军反舰弹道导弹数量" : 108\n    "美军西太平洋战机数量" : 430\n```
        
        输出:
        ```
        [
            "恒生科技指数在2025年第一季度累计涨幅达34.90%，显著超越同期沪深300指数（4.80%）和中证1000指数（16.63%）的涨幅。",
            "2025年1月13日至3月13日期间，南下资金日均净买入43亿港元，持续推动科技股行情。",
            "2025年第一季度约20%的科技股跑赢恒生科技指数，主要集中在互联网和人工智能（AI）领域。",
            "截至2025年3月13日，以美债利率衡量的港股ERP（股权风险溢价）回落至-1倍标准差区间，但结合中债利率的综合估值模型显示港股尚未进入极端估值状态。",
            "2025年3月26日美元兑港元汇率报7.7763，过去三个月（2025年1-3月）汇率波动幅度达0.3%，影响外资配置决策。",
            "2025年第一季度港股市场呈现结构性行情，被动型ETF资金占据主导地位，主动外资尚未大规模回流。",
            "pie\n    title 2027年中美台海军力对比预测\n    "中国海军水面舰艇" : 215\n    "美国印太舰队可部署舰艇" : 165\n    "解放军反舰弹道导弹数量" : 108\n    "美军西太平洋战机数量" : 430\n"
        ]
        ```

          
        # 用户输入
        输入: {input}

        输出: """
    )

judge_is_include = PromptTemplate(
    input_variables=["material", "content"],
    template="""
    你是一个蕴含关系判断助手。给定‘素材‘和‘正文‘，你需要判断‘正文’中所有的信息是否都在‘素材’中出现过。

    # 要求
    - 如果‘正文‘中所有的信息都在‘素材’中出现过，即素材蕴含正文，返回'1'，否则返回'0'，并简要说明原因。
    - 请按照json格式输出。

    # 示例
    ```json
    {{'result': '1', 'reason': 'xxx'}}
    ```
    ---
    # 用户输入
    素材: {material}

    正文: {content}

    输出: """
)

r1 = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), base_url= os.getenv("DEEPSEEK_BASE_URL")
)

def get_embed(input_):
    emb = client.embeddings.create(input = input_, model="text-embedding-ada-002").data
    return [e.embedding for e in emb]


def cal_hallu_score_with_llm(data):
    hallu_num = 0
    sentence_num = 0
    res = []
    for query, n in data.items():
        print(f"this section is {query}")
        # content = [i["content"] for i in n["useful_info"]]
        content = n['useful_info']
        answer = n["answer"]
        
        answer = answer.replace("\\n", "")
        answer = answer.replace("-", "")
        answer = answer.replace("`", "")
        re.sub(r'\s+', ' ', answer)

        
        prompt = chunk.invoke({"input":answer}).text
        messages = [
                {"role": "user", "content": prompt}
        ]

        llm_output =  r1.chat.completions.create(
                    model=model_id, messages=messages
            )
        all_sentences = llm_output.choices[0].message.content.strip()
        print("claim如下", all_sentences)

        
        all_sentences = re.findall(r"```(?:json)?\s*(.*)\s*```", all_sentences, re.DOTALL)[0]
        all_sentences = eval(all_sentences)

        sentence_num += len(all_sentences)


        # todo : 判断句子是否为幻觉: 如果句子中所有的信息都在content中出现过，那么就不是幻觉
        all_sentences_hallu_dic = {}
        t_res = {}
        for sentence in all_sentences:
            prompt = judge_is_include.invoke({"material":content, "content":sentence}).text
            messages = [
                {"role": "user", "content": prompt}
            ]
            llm_output =  r1.chat.completions.create(
                model=model_id, messages=messages
            )

            llm_output = llm_output.choices[0].message.content.strip()
            print(f"{sentence}是否为事实", llm_output)
            all_sentences_hallu_dic[sentence] =  eval(re.findall(r"```(?:json)?\s*(.*?)\s*```",llm_output, re.DOTALL)[0])['result']
            score_reason = eval(re.findall(r"```(?:json)?\s*(.*?)\s*```",llm_output, re.DOTALL)[0])['reason']
            
            t_res = {
                "section": query,
                "useful_info": content,
                "content": answer,
                "sentence": sentence,
                "is_hallu": 1- int(all_sentences_hallu_dic[sentence]),
                "hallu_reason": score_reason,
            }
            res.append(t_res)
    
    dic = {
        "section": [t['section'] for t in res],
        "useful_info": [t['useful_info'] for t in res],
        "content": [t['content'] for t in res],
        "sentence": [t['sentence'] for t in res],
        "is_hallu": [t['is_hallu'] for t in res],
        "hallu_reason": [t['hallu_reason'] for t in res],
    }
    
    return dic



def cal_hallu_score_with_llm_raw(data):
    """
        使用大模型评价 raw网页到useful info的 幻觉
    """

    hallu_num = 0
    sentence_num = 0
    res = []
    for n in data:
        # print(f"this section is {query}")
        # content = [i["content"] for i in n["useful_info"]]
        raw_info = [i['content'] for i in n['raw_info']]
        useful_info = n["useful_info"]   

        # todo : 判断句子是否为幻觉: 如果句子中所有的信息都在content中出现过，那么就不是幻觉
        all_sentences_hallu_dic = {}
        t_res = {}
        for sentence in useful_info:
            prompt = judge_is_include.invoke({"material":raw_info, "content":sentence['content']}).text
            messages = [
                {"role": "user", "content": prompt}
            ]
            llm_output =  r1.chat.completions.create(
                model=model_id, messages=messages
            )

            llm_output = llm_output.choices[0].message.content.strip()
            print(f"{sentence}是否为事实", llm_output)
            all_sentences_hallu_dic[sentence['content']] =  eval(re.findall(r"```(?:json)?\s*(.*?)\s*```",llm_output, re.DOTALL)[0])['result']
            score_reason = eval(re.findall(r"```(?:json)?\s*(.*?)\s*```",llm_output, re.DOTALL)[0])['reason']
            

            t_res = {
                "raw_info": raw_info,
                "useful_info": useful_info,
                "cur_useful_info": sentence['content'],
                "is_hallu": 1- int(all_sentences_hallu_dic[sentence['content']]),
                "hallu_reason": score_reason,
            }
            res.append(t_res)
    
    dic = {
        "raw_info": [t['raw_info'] for t in res],
        "useful_info": [t['useful_info'] for t in res],
        "cur_useful_info": [t['cur_useful_info'] for t in res],
        "is_hallu": [t['is_hallu'] for t in res],
        "hallu_reason": [t['hallu_reason'] for t in res],
    }
    
    return dic



def cal_hallu_score_with_embed(data):
    hallu_num = 0
    sentence_num = 0
    res = {}
    for query, n in data.items():
        print(f"this section is {query}")
        content = [i["content"] for i in n["useful_info"]]
        answer = n["answer"]
        
        answer = answer.replace("\\n", "")
        answer = answer.replace("-", "")
        answer = answer.replace("`", "")
        re.sub(r'\s+', ' ', answer)

        
        prompt = chunk.invoke({"input":answer}).text
        messages = [
                {"role": "user", "content": prompt}
        ]

        llm_output =  r1.chat.completions.create(
                    model=model_id, messages=messages
            )
        all_sentences = llm_output.choices[0].message.content.strip()
        print(all_sentences)

        
        all_sentences = re.findall(r"```(?:json)?\s*(.*?)\s*```", all_sentences, re.DOTALL)[0]
        all_sentences = eval(all_sentences)

        sentence_num += len(all_sentences)

        content_embed = get_embed(content)
        sentence_embed = get_embed(all_sentences)
        sim = cosine_similarity(sentence_embed, content_embed)
        sorted_indices = np.argsort(-sim)[:,:3]
        
        

        max_sim = np.max(sim, axis=1)
        indices = np.where(max_sim < 0.88)[0]
        
        hallu_num += len(indices)

        dict_to_insert = []
        for i in indices:
            print(f"sentence:{all_sentences[i]}")
            print(f"content : {[content[j] for j in sorted_indices[i]]}")
            dict_to_insert.append({all_sentences[i] : [content[j] for j in sorted_indices[i]]})
        
        res[query] = dict_to_insert

    res["hallu_score"] = 1.0 -  hallu_num / sentence_num
    return res




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process lines from an input file.")
    parser.add_argument("--input_file", default="/mnt/data/jinchang/smolagents/r1-reasoning-rag-v5_1/r1-reasoning-rag/src/eval/hallu/2017年中国会打台湾嘛_r1_raw_usefulinfo_0331.json", type=str, help="Path to the input file")
    parser.add_argument("--output_file", default='/mnt/data/jinchang/smolagents/r1-reasoning-rag-v5_1/r1-reasoning-rag/src/eval/hallu/2017年中国会打台湾嘛_r1_raw_usefulinfo_0331.xlsx', type=str, help="Path to the output file")
    parser.add_argument("--stage", default="usefulinfo", type=str, choices=['usefulinfo','raw'], help="which stage to eval")

    args = parser.parse_args()
    with open(args.input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    if args.stage == "usefulinfo":
        res = cal_hallu_score_with_llm(data)
    elif args.stage == "raw":
        res = cal_hallu_score_with_llm_raw(data)

    # with open(args.output_file, "w", encoding="utf-8") as file:
    #     json.dump(res, file, ensure_ascii=False, indent=4)
    from pyxcoder.xfile.xmain import FileMerger
    FileMerger(output_file=args.output_file).writer(res)
    

    


    

        
