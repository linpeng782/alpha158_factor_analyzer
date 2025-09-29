import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import partial
import warnings
import sys
import os

# 过滤警告信息
warnings.filterwarnings("ignore")

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factor_utils import *
from data_utils import *
from backtest import *
from batch_processing import batch_factor_analysis


def process_single_factor(
    factor_file_path: str,
    stock_universe: pd.DataFrame,
    cache_dir: str,
):
    """
    处理单个因子文件

    Args:
        factor_file_path: 因子文件路径
        stock_universe: 股票池DataFrame
        cache_dir: 缓存目录

    Returns:
        DataFrame: 包含IC指标和绩效指标的综合结果
    """
    # 从文件路径提取因子名称
    factor_name = factor_file_path.split("/")[-1].split("_")[1]

    print(f"\n{'='*60}")
    print(f"开始处理因子: {factor_name}")
    print(f"{'='*60}")

    # 对齐原始因子数据和股票池
    aligned_factor = align_raw_factor(factor_file_path, stock_universe)

    # 预处理: 新股、ST、停牌、涨停、标准化
    preprocessed_factor = preprocess_raw(aligned_factor, stock_universe, cache_dir)

    # 用市值过滤因子，取每日截面市值前70%的股票
    factor = filter_by_market_cap(preprocessed_factor, cache_dir)

    # 获取VWAP数据
    vwap_df = get_vwap_data(factor, cache_dir)

    # 计算IC、ICIR、t检验
    print("计算IC、ICIR、t检验...")
    ic_report_df, ic_direction, ic_values = calculate_ic(
        factor, vwap_df, rebalance_days=5, factor_name=factor_name
    )

    # 根据IC方向调整因子值
    adjusted_factor = factor * ic_direction

    # 生成买入队列及投资组合权重, rank_n = 100只股票
    buy_list = get_buy_list(adjusted_factor, rank_n=100)
    portfolio_weights = buy_list.div(buy_list.sum(axis=1), axis=0)
    portfolio_weights = portfolio_weights.shift(1).dropna(how="all")

    # 单因子回测
    print("单因子回测...")
    performance_result = backtest(portfolio_weights, vwap_df, rebalance_frequency=5)

    # 合并IC报告和回测绩效结果
    ic_report_reset = ic_report_df.reset_index()
    combined_result = pd.concat(
        [ic_report_reset, performance_result], axis=1
    ).set_index("factor_name")

    print(f"因子 {factor_name} 处理完成")
    return combined_result, ic_values


if __name__ == "__main__":
    # 设置多进程启动方式，避免一些平台相关问题
    mp.set_start_method("spawn", force=True)

    start_date = "2010-01-01"
    end_date = "2025-07-01"
    index_item = "000985.XSHG"

    DATA_PATH = "/Users/didi/DATA/alpha158"
    cache_dir = os.path.join(DATA_PATH, "cache")  # 缓存目录
    data_dir = os.path.join(DATA_PATH, "aggregated_factors")  # 因子数据目录
    save_dir = os.path.join(DATA_PATH, "factor_results")  # 结果保存目录

    print("获取股票池...")
    stock_universe = get_stock_universe(start_date, end_date, index_item, cache_dir)

    # 选择运行模式
    mode = "single"  # 可选: "single" 或 "batch"

    if mode == "single":
        # 单因子测试示例
        print("=== 单因子测试模式 ===")
        factor_file = os.path.join(data_dir, "factor_MA60_20250923.csv")
        result, ic_values = process_single_factor(
            factor_file, stock_universe, cache_dir
        )
        print("\n测试结果:")
        print(result)

    elif mode == "batch":
        # 批量处理模式
        print("=== 批量处理模式 ===")
        # 执行批量因子分析
        backtest_results, ic_values_df = batch_factor_analysis(
            data_dir,
            cache_dir,
            stock_universe,
            save_dir,
            n_processes=8,
        )
        print(f"\n批量处理完成，获得 {len(backtest_results)} 个因子的分析结果")
