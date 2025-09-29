import pandas as pd
import os
from factor_utils import get_price, get_vwap, get_trading_dates


def get_trading_days(stock_universe, cache_dir):

    try:
        trading_dates = pd.read_pickle(f"{cache_dir}/trading_dates.pkl")
        print(f"✅ 成功加载缓存的trading_dates")
        return trading_dates
    except:
        print(f"计算新的日历日期...")
        start_date = "2000-01-01"
        end_date = stock_universe.index.max()
        trading_dates = get_trading_dates(start_date, end_date)
        trading_dates = pd.DataFrame(trading_dates, columns=["datetime"])
        trading_dates.to_pickle(f"{cache_dir}/trading_dates.pkl")

    return trading_dates


def get_vwap_data(factor_df, cache_dir: str, top_pct=0.7) -> pd.DataFrame:

    try:
        vwap_df = pd.read_pickle(f"{cache_dir}/vwap_df_pct{int(top_pct*100)}.pkl")
        print(f"✅ 成功加载缓存的vwap_df")
        return vwap_df

    except:

        # 获取技术指标数据：成交额和成交量
        print("✅ 计算新的vwap_df...")
        stock_list = factor_df.columns.tolist()
        start_date = factor_df.index.min()
        end_date = factor_df.index.max()

        tech_list = ["total_turnover", "volume"]
        daily_tech = get_price(
            stock_list,
            start_date,
            end_date,
            fields=tech_list,
            adjust_type="post_volume",
            skip_suspended=False,
        ).sort_index()

        # 计算后复权VWAP（成交额/后复权调整后的成交量）
        post_vwap = daily_tech["total_turnover"] / daily_tech["volume"]

        # 获取未复权VWAP价格数据
        unadjusted_vwap = get_vwap(stock_list, start_date, end_date)

        # 转换为DataFrame并添加后复权VWAP
        vwap_df = pd.DataFrame(
            {"unadjusted_vwap": unadjusted_vwap, "post_vwap": post_vwap}
        )

        # 统一索引名称
        vwap_df.index.names = ["order_book_id", "datetime"]

        # 保存数据以备下次使用
        os.makedirs(cache_dir, exist_ok=True)
        vwap_df.to_pickle(f"{cache_dir}/vwap_df_pct{int(top_pct*100)}.pkl")

        return vwap_df


def optimize_ic_report(combined_ic_report):
    """
    优化IC报告：将负IC转换为正值，添加方向标识，并按IC降序排序

    Args:
        combined_ic_report: 合并后的IC报告DataFrame

    Returns:
        优化后的IC报告DataFrame
    """
    print("优化IC报告：将负IC转换为正值...")

    # 创建副本避免修改原数据
    optimized_report = combined_ic_report.copy()

    # 识别负IC的行
    negative_ic_mask = optimized_report["IC_mean"] < 0

    # 对负IC的行进行转换
    optimized_report.loc[negative_ic_mask, "IC_mean"] = abs(
        optimized_report.loc[negative_ic_mask, "IC_mean"]
    )
    optimized_report.loc[negative_ic_mask, "ICIR"] = abs(
        optimized_report.loc[negative_ic_mask, "ICIR"]
    )

    # 按优化后的策略年化收益降序排序
    optimized_report = optimized_report.sort_values("策略年化收益", ascending=False)

    return optimized_report


def check_raw_factor(raw_factor):

    print("原始因子数据前5行:")
    print(raw_factor.head())
    print(f"原始因子数据形状: {raw_factor.shape}")
    print(f"原始因子数据日期范围: {raw_factor.index.min()} 到 {raw_factor.index.max()}")

    # 统计每只股票的NaN数量
    nan_counts = raw_factor.isnull().sum()

    # 按NaN数量从高到低排序
    nan_counts_sorted = nan_counts.sort_values(ascending=False)

    print(f"\n每只股票的NaN数量统计 (总共{len(nan_counts_sorted)}只股票):")
    print("=" * 50)

    # 显示前20只NaN数量最多的股票
    print("NaN数量最多的前20只股票:")
    print(nan_counts_sorted.head(20))

    # 统计摘要信息
    print(f"\nNaN数量统计摘要:")
    print(f"最大NaN数量: {nan_counts_sorted.iloc[0]}")
    print(f"最小NaN数量: {nan_counts_sorted.iloc[-1]}")
    print(f"平均NaN数量: {nan_counts_sorted.mean():.2f}")
    print(f"中位数NaN数量: {nan_counts_sorted.median():.2f}")
    print(f"完全没有NaN的股票数量: {(nan_counts_sorted == 0).sum()}")

    return raw_factor


def align_raw_factor(data_dir, stock_universe):

    raw_factor = pd.read_csv(data_dir)
    raw_factor.set_index("date", inplace=True)
    raw_factor.index = pd.to_datetime(raw_factor.index)
    raw_factor.index.names = ["datetime"]

    # raw_factor = check_raw_factor(raw_factor)

    # 对齐：只保留两者都有的具体日期
    common_dates = raw_factor.index.intersection(stock_universe.index)
    aligned_factor = raw_factor.loc[common_dates]

    return aligned_factor
