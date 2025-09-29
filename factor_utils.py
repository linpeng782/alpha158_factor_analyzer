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


# 标准化处理
def standardize(df):
    df_standardize = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)
    return df_standardize


def filter_by_market_cap(preprocessed_factor, cache_dir, top_pct=0.7):

    print("过滤市值...")
    stock_list = preprocessed_factor.columns.tolist()
    start_date = preprocessed_factor.index.min()
    end_date = preprocessed_factor.index.max()

    # 对每个交易日，从有因子值的股票中选出市值排名前70%的股票
    try:
        market_cap_mask = pd.read_pickle(
            f"{cache_dir}/market_cap_mask_pct{int(top_pct*100)}.pkl"
        )
        print(f"✅ 成功加载缓存的market_cap_mask (前{int(top_pct*100)}%)")
    except:
        print(f"✅ 计算新的market_cap_mask (前{int(top_pct*100)}%)...")

        # 获取市值因子
        market_cap = execute_factor(
            Factor("market_cap_3"), stock_list, start_date, end_date
        )

        market_cap_mask_list = []

        for date in preprocessed_factor.index:
            factor_row = preprocessed_factor.loc[date]
            market_cap_row = market_cap.loc[date]

            mask = create_market_cap_mask(factor_row, market_cap_row, top_pct)
            market_cap_mask_list.append(mask)

        # 将所有mask合并成DataFrame
        market_cap_mask = pd.DataFrame(
            market_cap_mask_list, index=preprocessed_factor.index
        )

        os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
        market_cap_mask.to_pickle(
            f"{cache_dir}/market_cap_mask_pct{int(top_pct*100)}.pkl"
        )

    # 应用市值过滤
    factor = preprocessed_factor.mask(~market_cap_mask)

    # 标准化处理
    standardize_factor = standardize(factor)

    return standardize_factor


def create_market_cap_mask(factor_row, market_cap_row, top_pct=0.7):
    """
    对每一行（每个交易日）创建市值mask
    从有因子值的股票中选出市值排名前top_pct%的股票为True，其余为False

    :param factor_row: 因子数据的一行（一个交易日）
    :param market_cap_row: 市值数据的一行（一个交易日）
    :param top_pct: 保留的市值排名百分比，默认0.7（前70%）
    :return: 布尔mask，True表示保留该股票
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

    # 计算需要保留的股票数量（向上取整确保至少保留1只）
    n_stocks_to_keep = max(1, int(np.ceil(len(valid_market_cap) * top_pct)))

    # 从有因子值且有市值数据的股票中，选出市值最大的前top_pct%
    top_stocks = valid_market_cap.nlargest(n_stocks_to_keep).index

    # 创建mask：选中的股票为True，其余为False
    mask = pd.Series(False, index=factor_row.index)
    mask[top_stocks] = True

    return mask


def preprocess_raw(raw_factor, stock_universe, cache_dir):

    print("过滤新股、ST、停牌、涨停、完成标准化（暂未做行业市值中性化处理）...")
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

    # 标准化处理
    standardize_factor = standardize(factor)

    return standardize_factor


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
    ic_df,
    vwap_df,
    rebalance_days,
    Rank_IC=True,
    factor_name="",
):
    """
    计算因子IC
    :param ic_df: 因子数据 DataFrame
    :param rebalance_days: 换手周期（天数），可以是单个数字或列表
    :param Rank_IC: 是否使用排名IC
    :return: IC结果和报告
    """

    post_vwap = vwap_df["post_vwap"].unstack("order_book_id")

    # 未来一段收益股票的累计收益率计算
    future_returns = post_vwap.pct_change(rebalance_days).shift(-rebalance_days - 1)

    # 计算IC
    if Rank_IC:
        ic_values = ic_df.corrwith(future_returns, axis=1, method="spearman").dropna(
            how="all"
        )
    else:
        ic_values = ic_df.corrwith(future_returns, axis=1, method="pearson").dropna(
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
