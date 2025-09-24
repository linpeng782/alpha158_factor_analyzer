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

    # 过滤市值,取每日截面市值最大的前1000只股票
    factor = filter_by_market_cap(preprocessed_factor, save_dir)

    # 计算IC、ICIR、t检验
    ic_values, ic_report_df = calc_ic(
        factor, save_dir, rebalance_days=1, factor_name=factor_name
    )

    print(f"因子 {factor_name} 处理完成")
    return ic_report_df


def batch_factor_analysis(factors_dir, save_dir, start_date, end_date, index_item):
    """
    批量处理多个因子的IC分析
    """
    import glob

    print(f"开始批量因子分析...")
    print(f"因子目录: {factors_dir}")

    # 获取股票池（只需要计算一次）
    print("获取股票池...")
    stock_universe = INDEX_FIX(start_date, end_date, index_item)

    # 获取所有因子文件
    factor_files = glob.glob(f"{factors_dir}/factor_*.csv")
    print(f"发现 {len(factor_files)} 个因子文件")

    if len(factor_files) == 0:
        print("❌ 未发现任何因子文件！")
        return None

    # 存储所有因子的IC报告
    all_ic_reports = []

    # 逐个处理因子
    for i, factor_file in enumerate(factor_files, 1):
        factor_name = factor_file.split("/")[-1].split("_")[1]
        print(f"\n进度: [{i}/{len(factor_files)}] 处理因子: {factor_name}")

        try:
            ic_report_df = process_single_factor(factor_file, stock_universe, save_dir)
            all_ic_reports.append(ic_report_df)
        except Exception as e:
            print(f"❌ 因子 {factor_name} 处理失败: {str(e)}")
            continue

    if len(all_ic_reports) == 0:
        print("❌ 没有成功处理任何因子！")
        return None

    # 合并所有IC报告
    print(f"\n合并 {len(all_ic_reports)} 个因子的IC报告...")
    combined_ic_report = pd.concat(all_ic_reports, ignore_index=False)

    # 按IC_mean降序排序
    combined_ic_report = combined_ic_report.sort_values("IC_mean", ascending=False)

    # 保存合并后的报告
    report_path = f"{save_dir}/combined_ic_report.csv"
    combined_ic_report.to_csv(report_path)
    print(f"✅ 合并IC报告已保存到: {report_path}")

    # 显示汇总结果
    print(f"\n{'='*80}")
    print("因子IC分析汇总报告")
    print(f"{'='*80}")
    print(combined_ic_report)

    return combined_ic_report


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
