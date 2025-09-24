import pandas as pd


def check_raw_factor(data_dir):
    raw_factor = pd.read_csv(data_dir)
    raw_factor.set_index("date", inplace=True)
    raw_factor.index = pd.to_datetime(raw_factor.index)
    raw_factor.index.names = ["datetime"]

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

    raw_factor = check_raw_factor(data_dir)

    # 时间戳对齐：取两者的交集
    print("\n开始时间戳对齐...")

    # 进一步对齐：只保留两者都有的具体日期
    common_dates = raw_factor.index.intersection(stock_universe.index)
    aligned_factor = raw_factor.loc[common_dates]

    # 验证对齐结果
    if aligned_factor.shape[0] == stock_universe.shape[0]:
        print("✅ 时间戳对齐成功！")
    else:
        print("❌ 时间戳对齐失败，行数不匹配")

    return aligned_factor


def compare_df(factor_df, stock_universe):

    print("\n" + "=" * 60)
    print("比较factor_df和stock_universe的结构:")
    print("=" * 60)

    print(f"factor_df shape: {factor_df.shape}")
    print(f"stock_universe shape: {stock_universe.shape}")

    print(f"\nfactor_df index类型: {type(factor_df.index)}")
    print(f"stock_universe index类型: {type(stock_universe.index)}")

    print(f"\nfactor_df index范围: {factor_df.index.min()} 到 {factor_df.index.max()}")
    print(
        f"stock_universe index范围: {stock_universe.index.min()} 到 {stock_universe.index.max()}"
    )

    # 比较columns
    factor_cols = set(factor_df.columns)
    universe_cols = set(stock_universe.columns)

    print(f"\nfactor_df columns数量: {len(factor_cols)}")
    print(f"stock_universe columns数量: {len(universe_cols)}")

    common_cols = factor_cols & universe_cols
    factor_only = factor_cols - universe_cols
    universe_only = universe_cols - factor_cols

    print(f"共同的股票数量: {len(common_cols)}")
    print(f"只在factor_df中的股票数量: {len(factor_only)}")
    print(f"只在stock_universe中的股票数量: {len(universe_only)}")

    if len(factor_only) > 0:
        print(f"\n只在factor_df中的前10只股票: {list(factor_only)[:10]}")

    if len(universe_only) > 0:
        print(f"\n只在stock_universe中的前10只股票: {list(universe_only)[:10]}")

    # 比较index
    factor_dates = set(factor_df.index)
    universe_dates = set(stock_universe.index)

    common_dates = factor_dates & universe_dates
    factor_only_dates = factor_dates - universe_dates
    universe_only_dates = universe_dates - factor_dates

    print(f"\n共同的日期数量: {len(common_dates)}")
    print(f"只在factor_df中的日期数量: {len(factor_only_dates)}")
    print(f"只在stock_universe中的日期数量: {len(universe_only_dates)}")

    if len(factor_only_dates) > 0:
        print(f"只在factor_df中的前5个日期: {sorted(list(factor_only_dates))[:5]}")

    if len(universe_only_dates) > 0:
        print(
            f"只在stock_universe中的前5个日期: {sorted(list(universe_only_dates))[:5]}"
        )

    # 检查是否完全相同
    columns_identical = factor_cols == universe_cols
    index_identical = factor_dates == universe_dates

    print(f"\nColumns是否完全相同: {columns_identical}")
    print(f"Index是否完全相同: {index_identical}")
    print(f"结构是否完全匹配: {columns_identical and index_identical}")


def check_insufficient_stocks(factor, threshold=1000):
    """
    检查因子数据中有效股票数量少于阈值的日期
    """
    print(f"检查有效股票数量少于{threshold}的日期...")

    # 计算每日有效股票数量
    daily_counts = factor.notnull().sum(axis=1)

    # 找出少于阈值的日期
    insufficient_dates = daily_counts[daily_counts < threshold]

    if len(insufficient_dates) == 0:
        print(f"✅ 所有日期的有效股票数量都 >= {threshold}")
    else:
        print(f"❌ 发现 {len(insufficient_dates)} 个日期的有效股票数量 < {threshold}:")
        print("日期\t\t有效股票数量")
        print("-" * 30)
        for date, count in insufficient_dates.items():
            print(f"{date.strftime('%Y-%m-%d')}\t{count}")

    print(f"\n整体统计:")
    print(f"平均有效股票数量: {daily_counts.mean():.1f}")
    print(f"最小有效股票数量: {daily_counts.min()}")
    print(f"最大有效股票数量: {daily_counts.max()}")
    print(
        f"少于{threshold}只的日期占比: {len(insufficient_dates)/len(daily_counts)*100:.2f}%"
    )

    # 如果发现异常，也检查一下预处理前的数据
    if len(insufficient_dates) > 0:
        print("\n对比检查预处理前的数据:")
        check_insufficient_stocks(preprocessed_factor, threshold=1000)

        # 详细分析异常日期
        for date in insufficient_dates.index:
            print(f"\n详细分析异常日期: {date.strftime('%Y-%m-%d')}")
            print(f"对齐后因子有效股票数: {aligned_factor.loc[date].notnull().sum()}")
            print(
                f"预处理后有效股票数: {preprocessed_factor.loc[date].notnull().sum()}"
            )
            print(f"市值过滤后有效股票数: {factor.loc[date].notnull().sum()}")
