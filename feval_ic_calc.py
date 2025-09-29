import warnings
import sys
import os
import multiprocessing as mp

# 过滤警告信息
warnings.filterwarnings("ignore")

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factor_utils import get_stock_universe
from batch_processing import batch_factor_analysis, process_single_factor


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
            factor_file, stock_universe, cache_dir, save_dir
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
