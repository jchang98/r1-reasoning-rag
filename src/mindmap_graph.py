from nano_graphrag import GraphRAG, QueryParam
# from langchain.tools import BaseTool, StructuredTool, tool
import json
from dotenv import load_dotenv
import networkx as nx
load_dotenv()



class MindMap:
    def __init__(self, ini_content = "", working_dir="./local_mem" ):
        """
        Initialize the graph with a specified working directory.
        """
        self.working_dir = working_dir
        self.graph_func = GraphRAG(working_dir=self.working_dir)
        # 保存初始内容，但不立即插入
        self.ini_content = ini_content
        # 设置最小分块大小，避免网络为空
        self.graph_func.chunk_size = 512
        # 禁用聚类以避免空网络错误
        # self.graph_func.disable_clustering = True
        
    async def initialize(self):
        """
        异步初始化方法，用于插入内容
        """
        if self.ini_content:
            # 使用异步方法插入内容
            await self.graph_func.ainsert(self.ini_content)
        return self

    def get_entity_csv(self, path: str):
        # 读取 path下的文件，然后转化为csv 格式
        G = nx.read_graphml(path)

        # 计算入度并确定需要删除的节点（入度为0）
        in_degrees = dict(G.degree())
        nodes_to_remove = [n for n, d in in_degrees.items() if d == 0]
        G.remove_nodes_from(nodes_to_remove)

        # 构建CSV数据，确保ID连续
        entites_section_list = [["id", "entity", "type", "description"]]
        for i, n in enumerate(G.nodes(data=True)):
            entites_section_list.append(
                [
                    i,
                    n[0],
                    n[1].get("entity_type", "UNKNOWN"),
                    n[1].get("description", "UNKNOWN"),
                ]
            )
        entities_context = self.list_of_list_to_csv(entites_section_list)

        # 重新保存graphml文件
        nx.write_graphml(G, path)
        return entities_context
    
    def get_relation_csv(self, path: str):
        # 读取 path下的文件，然后转化为csv 格式
        G = nx.read_graphml(path)

        edges_datas = G.edges(data=True)
        edges_datas = list(edges_datas)

        relations_section_list = [
            ["id", "source", "target", "description", "weight"]
        ]
        for i, e in enumerate(edges_datas):
            relations_section_list.append(
                [
                    i,
                    e[0],
                    e[1],
                    e[2]["description"],
                    e[2]["weight"],
                ]
            )
        relations_context = self.list_of_list_to_csv(relations_section_list)
        return relations_context


    def list_of_list_to_csv(self, data: list[list]):
        return "\n".join(
            [",\t".join([str(data_dd) for data_dd in data_d]) for data_d in data]
        )
    
    def process_community_report(self, json_path="local_mem/kv_store_community_reports.json") -> str:
        """
        Read and process the community report JSON, returning the combined report string.
        """
        # Read JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Collect all report strings from each community
        all_reports = []
        for community_id, community in data.items():
            report_string = community.get("report_string", "")
            all_reports.append(f"Snippet {community_id}:\n{report_string}\n")
        
        # Combine all reports
        combined_reports = "\n".join(all_reports)
        return combined_reports

    async def graph_retrieval(self, query: str) -> None:
        """
        Insert content from './book.txt' into the graph index,
        then demonstrate both a global and a local query.
        """
        # # Perform a global graphrag search and print the result
        # print("Global search result:")
        # print(self.graph_func.query(query))
        
        # Perform a local graphrag search and print the result
        print("\nLocal graph search result:")
        res = await self.graph_func.aquery(query, 
                                    param=QueryParam(mode="local"))
        print(res)

        return res
    
    # async def graph_query(self, query: str) -> str:
    #     """
    #     Retrieve community reports by processing the local JSON store.
    #     """
    #     combined_report = self.process_community_report()
    #     print("\ncombined community report is:", combined_report)

    #     query = f"Answer the question:{query}\n\n based on the information:\n\n{combined_report}"

    #     return await self.process_community_report()

    async def __call__(self, query):
        """
        query the mind map knowledge graph and return the result
        """
        return await self.graph_retrieval(query)


async def main(test_text):
    import asyncio

    # 创建MindMap实例并使用测试文本初始化
    mind_map = MindMap(test_text)
    await mind_map.initialize()
    # 测试图检索功能
    # query_result = await mind_map.graph_retrieval("同花顺估值")
    # print(query_result)
    # return query_result


# Example usage:
if __name__ == "__main__":
    # # Create an instance of CreateGraph
    # graph_manager = MindMap(working_dir="./local_mem")
    
    # # Call graph_query which reads from './book.txt', inserts into GraphRAG, and prints query outputs
    # graph_manager.graph_query("your query here")
    
    # # Retrieve combined community report and print it
    # combined_report = graph_manager.graph_retrieval("dummy query")
    # print("\nCombined Community Report:")
    # print(combined_report)
    test_text = """
# A股短期投资市场环境分析（2025年3月）

2025年3月的A股市场环境呈现出显著的**政策驱动**特征。根据《2025年政府工作报告》，**“激发数字经济创新活力”**被明确列为培育新质生产力的核心方向，政策重点聚焦于**人工智能+**行动及智能终端产业升级。具体来看，新能源汽车、消费电子、智能制造装备等领域被多次提及，叠加**车辆购置税减免政策延续至2026年底**和新增**300亿元财政补贴**支持充电桩网络扩建，相关产业链的龙头企业（如动力电池、汽车电子、半导体）将直接受益。此外，中国人民银行于3月15日宣布下调存款准备金率50个基点至8.5%，释放长期流动性约**1.2万亿元**，此举不仅缓解了市场资金压力，更定向支持中小微企业与绿色低碳产业，为新能源电力运营商及环保设备制造商提供融资便利。  

从行业轮动趋势看，**耐用消费品**和**科技硬件**板块的短期机会尤为突出。2024年数据显示，中国新能源汽车销量达**1300万辆**，连续十年位居全球第一，而智能手机出货量同比增长4.7%至**2.8亿部**，规模化市场效应显著。2025年以旧换新补贴政策首次覆盖手机、平板等数码产品，预计带动消费电子产业链需求增长10%-15%。与此同时，**超长期特别国债3000亿元**的专项资金注入，进一步强化了家电、汽车等耐用消费品板块的修复预期。以智能家居和智能网联汽车为例，龙头企业凭借技术壁垒和政策红利，估值修复空间可达20%-30%。在科技领域，中国人工智能专利全球占比达**61.1%**，工业机器人新增装机量连续三年占全球50%以上，技术突破为**工业机器人**和**AI软件服务**企业带来估值溢价，建议关注研发投入占比超过10%的细分领域龙头。  

市场流动性指标与估值水平为保守型投资者提供了清晰的安全边际。截至2025年3月，A股动态市盈率中位数为**22.3倍**，其中**公用事业板块市盈率18.7倍**（低于历史10%分位）、**消费板块市盈率20.1倍**（低于历史15%分位），而创业板平均市盈率高达**45倍**，显示传统低估值板块更具防御性。结合宏观经济数据，2024年单位GDP能耗降幅超**3%**，可再生能源新增装机**3.7亿千瓦**，新能源电力运营商凭借稳定的现金流和高分红率（平均股息率4.2%）成为保守型配置的优选。此外，**智能制造装备**和**节能环保设备**制造商因政策扶持和技术迭代，未来3-6个月的盈利增速预期上调至15%-20%，风险收益比显著优于高估值成长股。  

```mermaid  
pie  
    title 2025年3月A股板块市盈率分布  
    "公用事业" : 18.7  
    "消费" : 20.1  
    "创业板" : 45  
```

| 政策工具               | 金额/幅度       | 受益领域               |
|------------------------|-----------------|------------------------|
| 存款准备金率下调       | 50个基点至8.5%  | 中小微企业、绿色产业   |
| 以旧换新专项补贴       | 3000亿元        | 家电、汽车、消费电子   |
| 车辆购置税减免延续     | 至2026年底      | 新能源汽车产业链       |
| 充电桩扩建补贴         | 300亿元         | 充电桩、动力电池回收   |

综上，当前A股市场的低风险机会集中于**政策明确支持**、**估值处于历史低位**且**现金流稳定**的板块。投资者可优先布局新能源汽车、消费电子、公用事业及智能制造装备领域的龙头企业，同时关注技术突破与政策催化共振下的细分行业alpha机会。
    """
    
    import asyncio
    asyncio.run(main(test_text))
