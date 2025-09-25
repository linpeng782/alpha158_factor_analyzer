import pandas as pd
import numpy as np
from typing import Union, Optional
import warnings
import sys
import os
import multiprocessing as mp
from functools import partial

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

    try:
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
        buy_list = get_buy_list(adjusted_factor, rank_n=100)
        portfolio_weights = buy_list.div(buy_list.sum(axis=1), axis=0)
        portfolio_weights = portfolio_weights.shift(1).dropna(how="all")

        # 单因子回测
        print("单因子回测...")
        performance_result = backtest(
            portfolio_weights, vwap_df, rebalance_frequency=20
        )

        # 合并IC报告和回测绩效结果
        # 将ic_report_df重置索引，使factor_name成为一列
        ic_report_reset = ic_report_df.reset_index()

        # 合并两个DataFrame
        combined_result = pd.concat(
            [ic_report_reset, performance_result], axis=1
        ).set_index("factor_name")

        print(f"因子 {factor_name} 处理完成")
        return combined_result

    except Exception as e:
        print(f"❌ 因子 {factor_name} 处理失败: {str(e)}")
        return None


def batch_factor_analysis(
    factors_dir, save_dir, start_date, end_date, index_item, n_processes=None
):
    """
    批量处理多个因子的IC分析（多进程版本）

    :param factors_dir: 因子文件目录
    :param save_dir: 缓存目录
    :param start_date: 开始日期
    :param end_date: 结束日期
    :param index_item: 指数代码
    :param n_processes: 进程数量，默认为CPU核心数-1
    """
    import glob
    import time

    print(f"开始批量因子分析（多进程版本）...")
    print(f"因子目录: {factors_dir}")

    # 获取股票池（只需要计算一次）
    print("获取股票池...")
    stock_universe = get_stock_universe(start_date, end_date, index_item, save_dir)

    # 获取所有因子文件
    factor_files = glob.glob(f"{factors_dir}/factor_*.csv")
    print(f"发现 {len(factor_files)} 个因子文件")

    # 确定进程数量
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)  # 留一个核心给系统

    print(f"使用 {n_processes} 个进程进行并行计算...")

    # 记录开始时间
    start_time = time.time()

    # 使用多进程池处理因子
    try:
        pool = mp.Pool(processes=n_processes)
        
        # 创建部分函数，固定stock_universe和save_dir参数
        process_func = partial(
            process_single_factor,
            stock_universe=stock_universe,
            save_dir=save_dir,
        )

        # 并行处理所有因子，使用imap获取实时进度
        results = []
        for i, result in enumerate(pool.imap(process_func, factor_files), 1):
            results.append(result)
            factor_name = factor_files[i - 1].split("/")[-1].split("_")[1]
            if result is not None:
                print(f"进度: [{i}/{len(factor_files)}] 因子 {factor_name} 完成")
            else:
                print(f"进度: [{i}/{len(factor_files)}] 因子 {factor_name} 失败")
                
    except KeyboardInterrupt:
        print("\n用户中断执行...")
        pool.terminate()
        pool.join()
        return None
    finally:
        # 确保进程池正确关闭
        pool.close()
        pool.join()

    # 过滤掉失败的结果
    all_factor_results = [result for result in results if result is not None]

    # 计算处理时间
    end_time = time.time()
    processing_time = end_time - start_time

    print(f"\n多进程处理完成！")
    print(f"成功处理: {len(all_factor_results)}/{len(factor_files)} 个因子")
    print(f"总耗时: {processing_time:.2f} 秒")
    print(f"平均每个因子: {processing_time/len(factor_files):.2f} 秒")

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


def batch_factor_analysis_single_thread(
    factors_dir, save_dir, start_date, end_date, index_item
):
    """
    批量处理多个因子的IC分析（单线程版本，用于调试）
    """
    import glob
    import time

    print(f"开始批量因子分析（单线程版本）...")
    print(f"因子目录: {factors_dir}")

    # 获取股票池（只需要计算一次）
    print("获取股票池...")
    stock_universe = get_stock_universe(start_date, end_date, index_item, save_dir)

    # 获取所有因子文件
    factor_files = glob.glob(f"{factors_dir}/factor_*.csv")
    print(f"发现 {len(factor_files)} 个因子文件")

    # 存储所有因子的完整分析结果（IC指标+绩效指标）
    all_factor_results = []

    # 记录开始时间
    start_time = time.time()

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

    # 计算处理时间
    end_time = time.time()
    processing_time = end_time - start_time

    print(f"\n单线程处理完成！")
    print(f"成功处理: {len(all_factor_results)}/{len(factor_files)} 个因子")
    print(f"总耗时: {processing_time:.2f} 秒")
    print(f"平均每个因子: {processing_time/len(factor_files):.2f} 秒")

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
    # 设置多进程启动方式，避免一些平台相关问题
    mp.set_start_method('spawn', force=True)

    data_dir = "/Users/didi/DATA/alpha158/large_factors"
    save_dir = "/Users/didi/DATA/alpha158/cache"
    start_date = "2010-01-01"
    end_date = "2025-07-01"
    index_item = "000985.XSHG"

    # 选择处理方式：True为多进程，False为单线程
    use_multiprocessing = True
    n_processes = 8  # 可以根据需要调整进程数量

    if use_multiprocessing:
        print("使用多进程模式进行批量因子分析...")
        combined_report = batch_factor_analysis(
            data_dir,
            save_dir,
            start_date,
            end_date,
            index_item,
            n_processes=n_processes,
        )
    else:
        print("使用单线程模式进行批量因子分析...")
        combined_report = batch_factor_analysis_single_thread(
            data_dir, save_dir, start_date, end_date, index_item
        )
