from src.agent import QAAgent
agent = QAAgent()
#agent.run("what is the capital city of Canada?")
# agent.run("帮我写一篇新能源汽车深度分析报告")
# agent.run("如果我想造一个哪吒同款风火轮，有可能实现吗？需要准备什么材料，要花多少钱？")
# agent.run("哪吒2爆火后，国漫IP能否构建自己的“漫威宇宙”")
agent.run("近期影响油价走势的因素", polish=False, polish_step=4)
