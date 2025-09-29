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

    # 预处理: 新股、ST、停牌、涨停
    preprocessed_factor = preprocess_raw(aligned_factor, stock_universe, cache_dir)

    # 用市值过滤因子，取每日截面市值最大的前1000只股票
    factor = filter_by_market_cap(preprocessed_factor, cache_dir)

    # 获取VWAP数据
    vwap_df = get_vwap_data(factor, cache_dir)

    # 计算IC、ICIR、t检验
    print("计算IC、ICIR、t检验...")
    ic_report_df, ic_direction, ic_values = calculate_ic(
        factor, cache_dir, rebalance_days=5, factor_name=factor_name
    )

    # 根据IC方向调整因子值
    adjusted_factor = factor * ic_direction

    # 生成买入队列及投资组合权重
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


def batch_factor_analysis(
    data_dir, cache_dir, stock_universe, save_dir, n_processes=None
):
    """
    批量处理多个因子的IC分析（多进程版本）

    :param data_dir: 数据文件目录
    :param cache_dir: 缓存目录
    :param stock_universe: 股票池DataFrame
    :param save_dir: 缓存目录
    :param n_processes: 进程数量，默认为CPU核心数-1
    """
    import glob
    import time

    print(f"开始批量因子分析（多进程版本）...")
    print(f"数据目录: {data_dir}")

    # 获取所有因子文件
    factor_files = glob.glob(f"{data_dir}/factor_*.csv")
    print(f"发现 {len(factor_files)} 个因子文件")

    # 确定进程数量
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)  # 留一个核心给系统

    print(f"使用 {n_processes} 个进程进行并行计算...")

    # 记录开始时间
    start_time = time.time()

    # 使用多进程池处理因子
    # 创建部分函数，固定stock_universe和cache_dir参数
    process_func = partial(
        process_single_factor,
        stock_universe=stock_universe,
        cache_dir=cache_dir,
    )

    # 使用上下文管理器确保资源正确释放
    backtest_results = []
    ic_values_list = []
    try:
        with mp.Pool(processes=n_processes) as pool:
            # 并行处理所有因子，使用imap获取实时进度
            for i, result in enumerate(pool.imap(process_func, factor_files), 1):
                factor_name = factor_files[i - 1].split("/")[-1].split("_")[1]
                if result is not None:
                    # result现在是一个元组 (combined_result, ic_values)
                    combined_result, ic_values = result
                    backtest_results.append(combined_result)
                    ic_values_list.append(ic_values)
                    print(f"进度: [{i}/{len(factor_files)}] 因子 {factor_name} 完成")
                else:
                    print(f"进度: [{i}/{len(factor_files)}] 因子 {factor_name} 失败")

            # 显式关闭和等待所有进程完成
            pool.close()
            pool.join()
    except Exception as e:
        print(f"多进程处理出错: {e}")
        return None

    # 过滤掉失败的结果
    all_backtest_results = [result for result in backtest_results if result is not None]
    all_ic_values = [ic_val for ic_val in ic_values_list if ic_val is not None]

    # 计算处理时间
    end_time = time.time()
    processing_time = end_time - start_time

    print(f"\n多进程处理完成！")
    print(f"成功处理: {len(all_backtest_results)}/{len(factor_files)} 个因子")
    print(f"总耗时: {processing_time:.2f} 秒")
    print(f"平均每个因子: {processing_time/len(factor_files):.2f} 秒")

    # 合并所有因子的分析结果并优化
    print(f"\n合并 {len(all_backtest_results)} 个因子的分析结果...")
    combined_backtest_results = optimize_ic_report(pd.concat(all_backtest_results))
    combined_ic_values = pd.concat(all_ic_values, axis=1)
    print(f"IC时间序列形状: {combined_ic_values.shape}")

    # 保存合并后的完整分析结果（添加时间戳）
    from datetime import datetime

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d")

    # 保存因子分析结果
    backtest_path = f"{save_dir}/combined_factor_analysis_{timestamp}.csv"
    ic_path = f"{save_dir}/combined_ic_values_{timestamp}.csv"

    print(f"✅ 合并因子分析、IC时间序列结果已保存到: {backtest_path, ic_path}")
    combined_backtest_results.to_csv(backtest_path, index=True)
    combined_ic_values.to_csv(ic_path, index=True)

    # 显示汇总结果
    print(f"\n{'='*80}")
    print("因子分析汇总报告（IC指标+绩效指标）")
    print(f"{'='*80}")
    print(combined_backtest_results)

    return combined_backtest_results, combined_ic_values


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
    mode = "batch"  # 可选: "single" 或 "batch"

    if mode == "single":
        # 单因子测试示例
        print("=== 单因子测试模式 ===")
        factor_file = os.path.join(data_dir, "factor_CORD30_20250923.csv")
        result, ic_values = process_single_factor(
            factor_file, stock_universe, cache_dir
        )
        print("\n测试结果:")
        print(result)
        print(f"\nIC时间序列形状: {ic_values.shape}")
        print("IC时间序列前5行:")
        print(ic_values.head())

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
