
from typing import TypedDict, List, Dict, Optional
from uuid import uuid4

class ResearchNode(TypedDict):
    id: str
    query: str
    research_goal: str
    learnings: List[str]
    childs: List['ResearchNode']
    urls: List[str]
    depth: int


def generate_tree_diagram(ancestor: ResearchNode, current_node: Optional[ResearchNode] = None) -> str:
    """生成带URL编号标注的研究路径图谱"""
    # 预收集所有唯一URL并建立编号映射
    url_registry = {}
    
    def collect_urls(node: ResearchNode):
        for url in node['urls']:
            if url not in url_registry:
                url_registry[url] = len(url_registry) + 1
        for child in node['childs']:
            collect_urls(child)
    
    collect_urls(ancestor)  # 构建URL注册表

    def build_tree_lines(
        node: ResearchNode,
        prefix: str = "",
        is_last: bool = True,
        is_root: bool = True
    ) -> List[str]:
        lines = []
        
        # 节点标题行
        current_marker = " ★" if node is current_node else ""
        depth_tag = f"[D{node['depth']+1}]" if not is_root else "[Origin]"
        connector = "└── " if is_last else "├── "
        main_line = f"{prefix}{connector}{depth_tag} {node['query']}{current_marker}"
        lines.append(main_line)
        
        # 研究目标区块
        goal_prefix = prefix + ("    " if is_last else "│   ")
        lines.append(f"{goal_prefix}├○ 研究目标: {trim_prompt(node['research_goal'], 60)}")
        
        # 关键认知区块
        learn_prefix = goal_prefix + "│   "
        if node['learnings']:
            lines.append(f"{learn_prefix}├• 关键认知")
            for i, learning in enumerate(node['learnings']):
                bullet = "└> " if i == len(node['learnings'])-1 else "├> "
                lines.append(f"{learn_prefix}│   {bullet}{trim_prompt(learning, 80)}")
        
        # 数据来源区块
        url_prefix = learn_prefix if node['learnings'] else goal_prefix
        if node['urls']:
            lines.append(f"{url_prefix}├◈ 数据来源")
            for i, url in enumerate(node['urls']):
                link_symbol = "└→ " if i == len(node['urls'])-1 else "├→ "
                citation_tag = f"[{url_registry[url]}] "
                lines.append(f"{url_prefix}│   {link_symbol}{citation_tag}{trim_prompt(url, 60)}")
        
        # 递归处理子节点
        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(node['childs']):
            is_last_child = i == len(node['childs'])-1
            lines.extend(build_tree_lines(child, child_prefix, is_last_child, False))
        
        return lines

    tree_lines = build_tree_lines(ancestor, is_root=True)
    return "\n".join([
        "研究路径图谱".center(80, "═"),
        *tree_lines,
        "═"*80,
        f"图谱统计｜认知节点: {count_nodes(ancestor)} | 唯一数据源: {len(url_registry)} | 最大深度: {get_max_depth(ancestor)}",
        "═"*80
    ])