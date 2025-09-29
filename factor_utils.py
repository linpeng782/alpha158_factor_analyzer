import scipy as sp
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from pathlib import Path
import os
from rqdatac import *
from rqfactor import *
from rqfactor import Factor
from rqfactor.extension import *

init("13522652015", "123456")
import rqdatac
from tqdm import *
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS",
    "PingFang SC",
    "Hiragino Sans GB",
    "STHeiti",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

import warnings


warnings.filterwarnings("ignore")


# 热力图
def hot_corr(name, ic_df, chart_dir):
    """
    :param name: 因子名称 -> list
    :param ic_df: ic序列表 -> dataframe
    :return fig: 热力图 -> plt
    """
    # 计算相关系数矩阵
    corr_matrix = ic_df[name].corr()
    
    # 动态调整图像大小，但设置合理的上下限
    n_factors = len(name)
    fig_size = max(12, min(30, n_factors * 0.3))  # 最小12，最大30
    
    plt.figure(figsize=(fig_size, fig_size))
    
    # 创建热力图，针对大量因子优化显示
    if n_factors > 50:
        # 因子数量多时，不显示数值标注，调整字体
        ax = sns.heatmap(
            corr_matrix, 
            cmap="Blues",  # 红-黄-蓝色谱，更容易区分
            center=0,  # 以0为中心
            square=True,
            linewidths=0.1,
            cbar_kws={"shrink": 0.8},
            xticklabels=True,
            yticklabels=True,
            annot=False  # 不显示数值标注
        )
        # 设置较小的字体
        plt.xticks(fontsize=6, rotation=90)
        plt.yticks(fontsize=6, rotation=0)
    else:
        # 因子数量少时，显示数值标注
        ax = sns.heatmap(
            corr_matrix,
            cmap="Blues",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            annot=True,
            fmt='.2f',
            annot_kws={'size': 8}
        )
        plt.xticks(fontsize=10, rotation=45)
        plt.yticks(fontsize=10, rotation=0)
    
    plt.title("Alpha158 因子IC相关性热力图", fontsize=16, pad=20)
    plt.tight_layout()
    
    # 确保目录存在
    os.makedirs(chart_dir, exist_ok=True)
    
    # 保存高分辨率图片
    save_path = f"{chart_dir}/Factors_IC_CORRELATION.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存
    
    print(f"✅ 因子IC热力图已保存到: {save_path}")
    print(f"📊 相关系数统计: 最大={corr_matrix.max().max():.3f}, 最小={corr_matrix.min().min():.3f}")
    
    # 输出高相关性因子对（相关系数>0.8）
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
    
    if high_corr_pairs:
        print(f"⚠️  发现 {len(high_corr_pairs)} 对高相关因子 (|相关系数|>0.8):")
        for factor1, factor2, corr_val in high_corr_pairs[:10]:  # 只显示前10对
            print(f"   {factor1} - {factor2}: {corr_val:.3f}")
        if len(high_corr_pairs) > 10:
            print(f"   ... 还有 {len(high_corr_pairs)-10} 对")
    else:
        print("✅ 未发现高相关因子对 (|相关系数|>0.8)")


def filter_by_market_cap(raw_factor, cache_dir, top_n=1000):

    print("过滤市值...")
    stock_list = raw_factor.columns.tolist()
    start_date = raw_factor.index.min()
    end_date = raw_factor.index.max()

    # 对每个交易日，从有因子值的股票中选出市值最大的前top_n只
    try:
        market_cap_mask = pd.read_pickle(f"{cache_dir}/market_cap_mask.pkl")
        print("✅ 成功加载缓存的market_cap_mask")
    except:
        print("✅ 计算新的market_cap_mask...")

        # 获取市值因子
        market_cap = execute_factor(
            Factor("market_cap_3"), stock_list, start_date, end_date
        )

        market_cap_mask_list = []

        for date in raw_factor.index:
            factor_row = raw_factor.loc[date]
            market_cap_row = market_cap.loc[date]

            mask = create_market_cap_mask(factor_row, market_cap_row, top_n)
            market_cap_mask_list.append(mask)

        # 将所有mask合并成DataFrame
        market_cap_mask = pd.DataFrame(market_cap_mask_list, index=raw_factor.index)

        os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
        market_cap_mask.to_pickle(f"{cache_dir}/market_cap_mask.pkl")

    # 应用市值过滤
    factor = raw_factor.mask(~market_cap_mask)

    return factor


def create_market_cap_mask(factor_row, market_cap_row, top_n=1000):
    """
    对每一行（每个交易日）创建市值mask
    从有因子值的股票中选出市值最大的前top_n只股票为True，其余为False
    """
    # 找到有因子值的股票（非NaN）
    valid_factor_stocks = factor_row.dropna().index

    if len(valid_factor_stocks) == 0:
        # 如果没有有效的因子数据，返回全False
        return pd.Series(False, index=factor_row.index)

    # 在有因子值的股票中，获取对应的市值数据
    valid_market_cap = market_cap_row[valid_factor_stocks].dropna()

    if len(valid_market_cap) == 0:
        # 如果没有有效的市值数据，返回全False
        return pd.Series(False, index=factor_row.index)

    # 从有因子值且有市值数据的股票中，选出市值最大的前top_n只
    top_stocks = valid_market_cap.nlargest(min(top_n, len(valid_market_cap))).index

    # 创建mask：选中的股票为True，其余为False
    mask = pd.Series(False, index=factor_row.index)
    mask[top_stocks] = True

    return mask


def preprocess_raw(raw_factor, stock_universe, cache_dir):

    print("过滤新股、ST、停牌、涨停...")
    stock_list = stock_universe.columns.tolist()
    date_list = stock_universe.index.tolist()

    try:
        combo_mask = pd.read_pickle(f"{cache_dir}/combo_mask.pkl")
        print("✅ 成功加载缓存的combo_mask")
    except:
        print("✅ 计算新的combo_mask...")
        # 新股过滤
        new_stock_filter = get_new_stock_filter(stock_list, date_list)
        # st过滤
        st_filter = get_st_filter(stock_list, date_list)
        # 停牌过滤
        suspended_filter = get_suspended_filter(stock_list, date_list)
        # 涨停过滤
        limit_up_filter = get_limit_up_filter(stock_list, date_list)

        # 合并过滤
        combo_mask = (
            new_stock_filter.astype(int)
            + st_filter.astype(int)
            + suspended_filter.astype(int)
            + limit_up_filter.astype(int)
            + (~stock_universe).astype(int)
        ) == 0

        os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
        combo_mask.to_pickle(f"{cache_dir}/combo_mask.pkl")

    factor = raw_factor.mask(~combo_mask)

    return factor


# 动态券池
def INDEX_FIX(start_date, end_date, index_item):
    """
    :param start_date: 开始日 -> str
    :param end_date: 结束日 -> str
    :param index_item: 指数代码 -> str
    :return index_fix: 动态因子值 -> unstack
    """

    index_fix = pd.DataFrame(
        {
            k: dict.fromkeys(v, True)
            for k, v in index_components(
                index_item, start_date=start_date, end_date=end_date
            ).items()
        }
    ).T

    index_fix.fillna(False, inplace=True)
    index_fix.index.names = ["datetime"]
    index_fix = index_fix.sort_index(axis=1)

    return index_fix


def get_stock_universe(start_date, end_date, index_item, cache_dir):
    """
    :param start_date: 开始日 -> str
    :param end_date: 结束日 -> str
    :param index_item: 指数代码 -> str
    :param cache_dir: 缓存目录 -> str
    :return stock_universe: 动态券池 -> unstack
    """

    try:
        stock_universe = pd.read_pickle(f"{cache_dir}/stock_universe.pkl")
        print("✅ 成功加载缓存的stock_universe")
    except:
        print("✅ 计算新的stock_universe...")
        stock_universe = INDEX_FIX(start_date, end_date, index_item)
        os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
        stock_universe.to_pickle(f"{cache_dir}/stock_universe.pkl")

    return stock_universe


# 新股过滤
def get_new_stock_filter(stock_list, datetime_period, newly_listed_threshold=252):
    """
    :param stock_list: 股票队列 -> list
    :param datetime_period: 研究周期 -> list
    :param newly_listed_threshold: 新股日期阈值 -> int
    :return newly_listed_window: 新股过滤券池 -> unstack
    """

    datetime_period_tmp = datetime_period.copy()
    # 多添加一天
    datetime_period_tmp += [
        pd.to_datetime(get_next_trading_date(datetime_period[-1], 1))
    ]
    # 获取上市日期
    listed_datetime_period = [instruments(stock).listed_date for stock in stock_list]
    # 获取上市后的第252个交易日（新股和老股的分界点）
    newly_listed_window = pd.Series(
        index=stock_list,
        data=[
            pd.to_datetime(get_next_trading_date(listed_date, n=newly_listed_threshold))
            for listed_date in listed_datetime_period
        ],
    )
    # 防止分割日在研究日之后，后续填充不存在
    for k, v in enumerate(newly_listed_window):
        if v > datetime_period_tmp[-1]:
            newly_listed_window.iloc[k] = datetime_period_tmp[-1]

    # 标签新股，构建过滤表格
    newly_listed_window.index.names = ["order_book_id"]
    newly_listed_window = newly_listed_window.to_frame("date")
    newly_listed_window["signal"] = True
    newly_listed_window = (
        newly_listed_window.reset_index()
        .set_index(["date", "order_book_id"])
        .signal.unstack("order_book_id")
        .reindex(index=datetime_period_tmp)
    )
    newly_listed_window = newly_listed_window.shift(-1).bfill().fillna(False).iloc[:-1]

    return newly_listed_window


# st过滤（风险警示标的默认不进行研究）
def get_st_filter(stock_list, date_list):
    """
    :param stock_list: 股票池 -> list
    :param date_list: 研究周期 -> list
    :return st_filter: st过滤券池 -> unstack
    """

    # 当st时返回1，非st时返回0
    st_filter = is_st_stock(stock_list, date_list[0], date_list[-1]).reindex(
        columns=stock_list, index=date_list
    )
    st_filter = st_filter.shift(-1).ffill()

    return st_filter


# 停牌过滤 （无法交易）
def get_suspended_filter(stock_list, date_list):
    """
    :param stock_list: 股票池 -> list
    :param date_list: 研究周期 -> list
    :return suspended_filter: 停牌过滤券池 -> unstack
    """

    # 当停牌时返回1，非停牌时返回0
    suspended_filter = is_suspended(stock_list, date_list[0], date_list[-1]).reindex(
        columns=stock_list, index=date_list
    )
    suspended_filter = suspended_filter.shift(-1).ffill()

    return suspended_filter


# 涨停过滤 （开盘无法买入）
def get_limit_up_filter(stock_list, date_list):
    """
    :param stock_list: 股票池 -> list
    :param date_list: 研究周期 -> list
    :return limit_up_filter: 涨停过滤券池 -> unstack∏
    """

    price = get_price(
        stock_list,
        date_list[0],
        date_list[-1],
        adjust_type="none",
        fields=["open", "limit_up"],
    )
    limit_up_mask = (
        (price["open"] == price["limit_up"])
        .unstack("order_book_id")
        .shift(-1)
        .fillna(False)
    )

    return limit_up_mask


# 单因子检验
def calculate_ic(
    df,
    cache_dir,
    rebalance_days,
    Rank_IC=True,
    factor_name="",
):
    """
    计算因子IC
    :param df: 因子数据 DataFrame
    :param rebalance_days: 换手周期（天数），可以是单个数字或列表
    :param Rank_IC: 是否使用排名IC
    :return: IC结果和报告
    """

    vwap_df = pd.read_pickle(f"{cache_dir}/vwap_df.pkl")
    post_vwap = vwap_df["post_vwap"].unstack("order_book_id")

    # 未来一段收益股票的累计收益率计算
    future_returns = post_vwap.pct_change(rebalance_days).shift(-rebalance_days - 1)

    # 计算IC
    if Rank_IC:
        ic_values = df.corrwith(future_returns, axis=1, method="spearman").dropna(
            how="all"
        )
    else:
        ic_values = df.corrwith(future_returns, axis=1, method="pearson").dropna(
            how="all"
        )

    # t检验 单样本
    t_stat, _ = stats.ttest_1samp(ic_values, 0)

    # 计算IC方向：根据IC均值的正负确定因子方向
    ic_mean = ic_values.mean()
    ic_direction = 1 if ic_mean >= 0 else -1

    # 因子报告
    ic_report = {
        "factor_name": factor_name,
        "rebalance_days": rebalance_days,
        "IC_mean": round(ic_mean, 4),
        "IC_std": round(ic_values.std(), 4),
        "ICIR": round(ic_mean / ic_values.std(), 4),
        "IC_>0": round(len(ic_values[ic_values > 0].dropna()) / len(ic_values), 4),
        "ABS_IC_>2%": round(
            len(ic_values[abs(ic_values) > 0.02].dropna()) / len(ic_values), 4
        ),
        "t_statistic": round(t_stat, 4),
        "ic_direction": ic_direction,
    }

    # 转换为DataFrame格式
    ic_report_df = pd.DataFrame([ic_report]).set_index("factor_name")
    ic_values = pd.DataFrame(ic_values, columns=[factor_name])

    print(ic_report_df)

    return ic_report_df, ic_direction, ic_values
