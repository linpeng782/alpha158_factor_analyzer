import pandas as pd
import numpy as np
from typing import Union, Optional
import warnings
import sys
import os

warnings.filterwarnings("ignore")

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factor_utils import *
from data_utils import *
from backtest import *


def process_single_factor(factor_file_path, stock_universe, save_dir):
    """
    处理单个因子文件
    """
    # 从文件名提取因子名称
    factor_name = factor_file_path.split("/")[-1].split("_")[1]  # 提取BETA10

    print(f"\n{'='*60}")
    print(f"开始处理因子: {factor_name}")
    print(f"{'='*60}")

    # 对齐原始因子数据和股票池
    aligned_factor = align_raw_factor(factor_file_path, stock_universe)

    # 预处理: 新股、ST、停牌、涨停
    preprocessed_factor = preprocess_raw(aligned_factor, stock_universe, save_dir)

    # 用市值过滤因子，取每日截面市值最大的前1000只股票
    factor = filter_by_market_cap(preprocessed_factor, save_dir)

    # 获取VWAP数据
    vwap_df = get_vwap_data(factor, save_dir)

    # 计算IC、ICIR、t检验
    print("计算IC、ICIR、t检验...")
    ic_report_df, ic_direction = calculate_ic(
        factor, save_dir, rebalance_days=5, factor_name=factor_name
    )

    # 根据IC方向调整因子值
    adjusted_factor = factor * ic_direction

    # 生成买入队列及投资组合权重
    buy_list = get_buy_list(adjusted_factor, rank_n=200)
    portfolio_weights = buy_list.div(buy_list.sum(axis=1), axis=0)
    portfolio_weights = portfolio_weights.shift(1).dropna(how="all")

    # 单因子回测
    print("单因子回测...")
    performance_result = backtest(portfolio_weights, vwap_df, rebalance_frequency=5)

    # 合并IC报告和回测绩效结果
    # 将ic_report_df重置索引，使factor_name成为一列
    ic_report_reset = ic_report_df.reset_index()

    # 合并两个DataFrame
    combined_result = pd.concat(
        [ic_report_reset, performance_result], axis=1
    ).set_index("factor_name")

    print(f"因子 {factor_name} 处理完成")
    return combined_result


def batch_factor_analysis(factors_dir, save_dir, start_date, end_date, index_item):
    """
    批量处理多个因子的IC分析
    """
    import glob

    print(f"开始批量因子分析...")
    print(f"因子目录: {factors_dir}")

    # 获取股票池（只需要计算一次）
    print("获取股票池...")
    stock_universe = get_stock_universe(start_date, end_date, index_item, save_dir)

    # 获取所有因子文件
    factor_files = glob.glob(f"{factors_dir}/factor_*.csv")
    print(f"发现 {len(factor_files)} 个因子文件")

    # 存储所有因子的完整分析结果（IC指标+绩效指标）
    all_factor_results = []

    # 逐个处理因子
    for i, factor_file in enumerate(factor_files, 1):
        factor_name = factor_file.split("/")[-1].split("_")[1]
        print(f"\n进度: [{i}/{len(factor_files)}] 处理因子: {factor_name}")

        try:
            combined_result = process_single_factor(
                factor_file, stock_universe, save_dir
            )
            all_factor_results.append(combined_result)
        except Exception as e:
            print(f"❌ 因子 {factor_name} 处理失败: {str(e)}")
            continue

    if len(all_factor_results) == 0:
        print("❌ 没有成功处理任何因子！")
        return None

    # 合并所有因子的分析结果
    print(f"\n合并 {len(all_factor_results)} 个因子的分析结果...")
    combined_results = pd.concat(all_factor_results)

    # 保存合并后的完整分析结果
    report_path = f"{save_dir}/combined_factor_analysis.csv"
    combined_results.to_csv(report_path, index=True)
    print(f"✅ 合并因子分析结果已保存到: {report_path}")

    # 显示汇总结果
    print(f"\n{'='*80}")
    print("因子分析汇总报告（IC指标+绩效指标）")
    print(f"{'='*80}")
    print(combined_results)

    return combined_results


if __name__ == "__main__":

    factors_dir = "/Users/didi/DATA/alpha158/large_factors"
    save_dir = "/Users/didi/DATA/alpha158/cache"
    start_date = "2010-01-01"
    end_date = "2025-07-01"
    index_item = "000985.XSHG"

    # 批量处理所有因子
    combined_report = batch_factor_analysis(
        factors_dir, save_dir, start_date, end_date, index_item
    )
