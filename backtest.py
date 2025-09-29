import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from rqdatac import *
from rqfactor import *
from rqfactor import Factor
from rqfactor.extension import *
from datetime import datetime

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


# 创建分离式策略报告：收益曲线图 + 绩效指标表
from matplotlib import rcParams
import os

# 设置中文字体
rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
import warnings

warnings.filterwarnings("ignore")


def get_buy_list(df, top_type="rank", rank_n=100, quantile_q=0.8):
    """
    :param df: 因子值 -> dataframe/unstack
    :param top_type: 选择买入队列方式，从['rank','quantile']选择一种方式 -> str
    :param rank_n: 值最大的前n只的股票 -> int
    :param quantile_q: 值最大的前n分位数的股票 -> float
    :return df: 买入队列 -> dataframe/unstack
    """

    if top_type == "rank":
        df = df.rank(axis=1, ascending=False) <= rank_n
    elif top_type == "quantile":
        df = df.sub(df.quantile(quantile_q, axis=1), axis=0) > 0
    else:
        print("select one from ['rank','quantile']")

    df = df.astype(int)
    df = df.replace(0, np.nan).dropna(how="all", axis=1)

    return df


def calc_transaction_fee(
    transaction_value, min_transaction_fee, sell_cost_rate, buy_cost_rate
):
    """
    计算单笔交易的手续费

    :param transaction_value: 交易金额（正数为买入，负数为卖出）
    :param min_transaction_fee: 最低交易手续费
    :param sell_cost_rate: 卖出成本费率
    :param buy_cost_rate: 买入成本费率
    :return: 交易手续费
    """
    if pd.isna(transaction_value) or transaction_value == 0:
        return 0  # 无交易时手续费为0
    elif transaction_value < 0:  # 卖出交易（负数）
        fee = -transaction_value * sell_cost_rate  # 卖出手续费：印花税 + 过户费 + 佣金
    else:  # 买入交易（正数）
        fee = transaction_value * buy_cost_rate  # 买入手续费：过户费 + 佣金

    # 应用最低手续费限制
    return max(fee, min_transaction_fee)  # 返回实际手续费和最低手续费中的较大值


def calculate_target_holdings(
    target_weights, available_cash, stock_prices, min_trade_units, sell_cost_rate
):
    """
    计算目标持仓数量

    :param target_weights: 目标权重 Series
    :param available_cash: 可用资金
    :param stock_prices: 股票价格 Series
    :param min_trade_units: 最小交易单位 Series
    :param sell_cost_rate: 卖出成本费率（用于预留手续费）
    :return: 目标持仓数量 Series
    """
    # 按权重分配资金
    allocated_cash = target_weights * available_cash

    # 计算调整后价格（预留卖出手续费）
    adjusted_prices = stock_prices * (1 + sell_cost_rate)

    # 计算可购买的最小交易单位数量（向下取整）
    units_to_buy = allocated_cash / adjusted_prices // min_trade_units

    # 转换为实际股数
    target_holdings = units_to_buy * min_trade_units

    return target_holdings


def get_stock_vwap(vwap_data, adjust):

    # 根据复权类型返回相应的价格数据
    if adjust == "post":
        adjusted_price = vwap_data["post_vwap"].unstack("order_book_id")
        return adjusted_price
    elif adjust == "none":
        unadjusted_price = vwap_data["unadjusted_vwap"].unstack("order_book_id")
        return unadjusted_price


def backtest(
    portfolio_weights,
    vwap_df,
    rebalance_frequency=20,
    initial_capital=10000 * 10000,
    stamp_tax_rate=0.0005,
    transfer_fee_rate=0.0001,
    commission_rate=0.0002,
    min_transaction_fee=5,
    cash_annual_yield=0.02,
    backtest_start_date=None,
):
    """
    量化策略回测框架

    :param portfolio_weights: 投资组合权重矩阵 -> DataFrame
    :param rebalance_frequency: 调仓频率（天数） -> int
    :param initial_capital: 初始资本 -> float
    :param stamp_tax_rate: 印花税率 -> float
    :param transfer_fee_rate: 过户费率 -> float
    :param commission_rate: 佣金率 -> float
    :param min_transaction_fee: 最低交易手续费 -> float
    :param cash_annual_yield: 现金年化收益率 -> float
    :param start_date: 回测起始日期，格式为'YYYY-MM-DD'或None（从数据开始日期开始） -> str or None
    :return: 账户历史记录 -> DataFrame
    """

    # =========================== 基础参数初始化 ===========================
    # 保存初始资本备份，用于最后的统计计算
    cash = initial_capital
    # 初始化历史持仓，第一次调仓时为0
    previous_holdings = 0
    # 买入成本费率：过户费 + 佣金
    buy_cost_rate = transfer_fee_rate + commission_rate
    # 卖出成本费率：印花税 + 过户费 + 佣金
    sell_cost_rate = stamp_tax_rate + transfer_fee_rate + commission_rate
    # 现金账户日利率（年化收益率转换为日收益率）
    daily_cash_yield = (1 + cash_annual_yield) ** (1 / 252) - 1
    # 筛选出回测期间的交易日
    all_signal_dates = portfolio_weights.index.tolist()
    # 处理起始日期参数
    if backtest_start_date is not None:
        # 将字符串日期转换为pandas Timestamp
        start_timestamp = pd.to_datetime(backtest_start_date)
        # 筛选出大于等于起始日期的信号日期
        filtered_signal_dates = [
            date for date in all_signal_dates if date >= start_timestamp
        ]
        if not filtered_signal_dates:
            raise ValueError(f"指定的起始日期 {backtest_start_date} 大于所有数据日期")
        all_signal_dates = filtered_signal_dates
        print(
            f"回测起始日期: {backtest_start_date}, 实际开始日期: {all_signal_dates[0].strftime('%Y-%m-%d')}"
        )
    # =========================== 数据结构初始化 ===========================
    # 创建账户历史记录表，索引为回测期间的交易日
    account_history = pd.DataFrame(
        index=pd.Index(all_signal_dates),  # 使用筛选后的日期范围
        columns=["total_account_asset", "holding_market_cap", "cash_account"],
    )
    # 获取所有股票的未复权价格数据
    unadjusted_prices = get_stock_vwap(vwap_df, "none")
    # 获取所有股票的后复权价格数据
    adjusted_prices = get_stock_vwap(vwap_df, "post")
    # 获取每只股票的最小交易单位（通常为100股）
    min_trade_units = pd.Series(
        dict([(stock, 100) for stock in portfolio_weights.columns.tolist()])
    )

    # 生成调仓日期列表：每 rebalance_frequency 天调仓一次，最后一天也被包含在调仓日中
    rebalance_dates = sorted(
        set(all_signal_dates[::rebalance_frequency] + [all_signal_dates[-1]])
    )

    # =========================== 开始逐期调仓循环 ===========================
    for i in tqdm(range(0, len(rebalance_dates) - 1)):

        rebalance_date = rebalance_dates[i]  # 当前调仓日期
        next_rebalance_date = rebalance_dates[i + 1]  # 下一个调仓日期

        # if rebalance_date == pd.Timestamp("2016-05-27"):
        #     breakpoint()

        # =========================== 获取当前调仓日的目标权重 ===========================
        # 获取当前调仓日的目标权重，并删除缺失值
        target_weights = portfolio_weights.loc[rebalance_date].dropna()
        # 获取目标股票列表
        target_stocks = target_weights.index.tolist()

        # =========================== 计算目标持仓数量 ===========================
        target_holdings = calculate_target_holdings(
            target_weights,
            cash,
            unadjusted_prices.loc[rebalance_date, target_stocks],
            min_trade_units.loc[target_stocks],
            sell_cost_rate,
        )

        # =========================== 仓位变动计算 ===========================
        ## 步骤1：计算持仓变动量（目标持仓 - 历史持仓）
        # fill_value=0 确保新增股票（历史持仓为空）和清仓股票（目标持仓为空）都能正确计算
        holdings_change_raw = target_holdings.sub(previous_holdings, fill_value=0)

        ## 步骤2：过滤掉无变动的股票（变动量为0的股票,用np.nan代替）
        holdings_change_filtered = holdings_change_raw.replace(0, np.nan)

        ## 步骤3：删除NaN,获取最终的交易执行列表
        trades_to_execute = holdings_change_filtered.dropna()

        # 获取当前调仓日的所有股票未复权价格
        current_prices = unadjusted_prices.loc[rebalance_date]

        # =========================== 计算交易成本 ===========================
        # 计算总交易成本：交易金额 = 价格 * 交易股数，根据交易金额计算手续费
        total_transaction_cost = (
            (current_prices * trades_to_execute)
            .apply(
                lambda x: calc_transaction_fee(
                    x, min_transaction_fee, sell_cost_rate, buy_cost_rate
                )
            )
            .sum()
        )

        # =========================== 价格复权调整 ===========================
        # 从调仓日到下一调仓日的后复权价格
        period_adj_prices = adjusted_prices.loc[rebalance_date:next_rebalance_date]
        # 调仓日的后复权价格(基准)
        base_adj_prices = adjusted_prices.loc[rebalance_date]
        # 价格变动倍数
        price_multipliers = period_adj_prices.div(base_adj_prices, axis=1)
        # 模拟未复权价格
        simulated_prices = price_multipliers.mul(current_prices, axis=1).dropna(
            axis=1, how="all"
        )

        # =========================== 计算投资组合市值 ===========================
        # 投资组合市值 = 每只股票的(模拟未复权价格 * 持仓数量)的总和
        # 处理价格缺失的情况：当价格为NaN时，使用前一日价格填充
        simulated_prices_filled = simulated_prices.ffill()

        # 按日计算投资组合市值（忽略NaN值）
        portfolio_market_value = (simulated_prices_filled * target_holdings).sum(axis=1)

        # =========================== 计算现金账户余额 ===========================
        # 初始现金余额 = 可用资金 - 交易成本 - 初始投资金额
        initial_cash_balance = (
            cash - total_transaction_cost - portfolio_market_value.loc[rebalance_date]
        )

        # 计算期间现金账户的复利增长（按日计息）
        cash_balance = pd.Series(
            [
                initial_cash_balance
                * ((1 + daily_cash_yield) ** (day + 1))  # 复利计息公式
                for day in range(0, len(portfolio_market_value))
            ],  # 对每一天计算
            index=portfolio_market_value.index,
        )  # 使用相同的日期索引

        # =========================== 计算账户总资产 ===========================
        total_portfolio_value = (
            portfolio_market_value + cash_balance
        )  # 总资产 = 持仓市值 + 现金余额

        # =========================== 更新历史数据为下一次调仓做准备 ===========================
        previous_holdings = target_holdings  # 更新历史持仓为当前目标持仓
        cash = total_portfolio_value.loc[
            next_rebalance_date
        ]  # 更新可用资金为下一调仓日的账户总值

        # =========================== 保存账户历史记录 ===========================
        # 将当前期间的账户数据保存到历史记录中（保留2位小数）
        account_history.loc[
            rebalance_date:next_rebalance_date, "total_account_asset"
        ] = round(total_portfolio_value, 2)
        account_history.loc[
            rebalance_date:next_rebalance_date, "holding_market_cap"
        ] = round(portfolio_market_value, 2)
        account_history.loc[rebalance_date:next_rebalance_date, "cash_account"] = round(
            cash_balance, 2
        )

    # =========================== 添加初始日记录并排序 ===========================
    # 在第一个交易日之前添加初始资本记录
    initial_date = pd.to_datetime(
        get_previous_trading_date(account_history.index.min(), 1)
    )
    account_history.loc[initial_date] = [
        initial_capital,
        0,
        initial_capital,
    ]  # [总资产, 持仓市值, 现金余额]
    account_history = account_history.sort_index()  # 按日期排序

    # =========================== 做一个简单的回测 ===========================
    rf = 0.03
    performance = account_history["total_account_asset"]
    daily_returns = performance.pct_change().dropna(how="all")

    cumulative_returns = (1 + daily_returns).cumprod()

    daily_pct_change = cumulative_returns.pct_change().dropna()

    strategy_final_return = cumulative_returns.iloc[-1] - 1
    strategy_annualized_return = (strategy_final_return + 1) ** (
        252 / len(cumulative_returns)
    ) - 1

    # 波动率
    strategy_volatility = daily_pct_change.std() * np.sqrt(252)

    # 夏普
    strategy_sharpe = (strategy_annualized_return - rf) / strategy_volatility

    result = pd.DataFrame(
        {
            "策略累计收益": [round(strategy_final_return, 4)],
            "策略年化收益": [round(strategy_annualized_return, 4)],
            "夏普比率": [round(strategy_sharpe, 4)],
        }
    )

    return result
