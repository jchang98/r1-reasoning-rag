import pandas as pd
import numpy as np
from datetime import datetime

# 今天的日期（根据题目要求）
today = datetime(2025, 4, 3).date()

# 用户提供的DataFrame数据
data = {
    '股票简称': ['同花顺'],
    '股票代码': ['300033.SZ'],
    '向下有效突破均线': ['否'],
    'ADTM动态买卖气指标': [-0.171154],
    'B3612三减六日乖离': [-8.06333]
}
df = pd.DataFrame(data)

# 分析函数
def analyze_stock_technical(df):
    print("\n=== 股票技术指标分析报告 ===")
    print(f"分析日期: {today}")
    print(f"股票简称: {df['股票简称'].values[0]}")
    print(f"股票代码: {df['股票代码'].values[0]}\n")
    
    # 分析向下有效突破均线
    ma_breakthrough = df['向下有效突破均线'].values[0]
    print(f"1. 向下有效突破均线: {ma_breakthrough}")
    if ma_breakthrough == '是':
        print("   → 警告: 该股票价格已向下突破均线，可能进入下跌趋势")
    else:
        print("   → 当前价格未向下突破均线，均线支撑仍然有效")
    
    # 分析ADTM动态买卖气指标
    adtm = df['ADTM动态买卖气指标'].values[0]
    print(f"\n2. ADTM动态买卖气指标: {adtm:.6f}")
    if adtm > 0.5:
        print("   → 强烈买入信号: 市场买气旺盛")
    elif adtm > 0:
        print("   → 买入信号: 市场买气占优")
    elif adtm > -0.5:
        print("   → 卖出信号: 市场卖气占优")
    else:
        print("   → 强烈卖出信号: 市场卖气强烈")
    
    # 分析B3612三减六日乖离
    b3612 = df['B3612三减六日乖离'].values[0]
    print(f"\n3. B3612三减六日乖离: {b3612:.6f}")
    if b3612 > 0:
        print("   → 短期趋势强于中期趋势，可能有上涨机会")
    else:
        print("   → 短期趋势弱于中期趋势，需谨慎操作")
    
    # 综合评估
    print("\n4. 综合评估:")
    if ma_breakthrough == '否' and adtm > -0.5 and b3612 > -5:
        print("   → 中性偏多: 技术指标显示该股票目前没有明显下跌风险")
    elif ma_breakthrough == '是' or adtm < -0.5 or b3612 < -5:
        print("   → 看空信号: 多个技术指标显示该股票可能存在下跌风险")
    else:
        print("   → 中性: 技术指标信号不一，建议观望")

# 执行分析
analyze_stock_technical(df)