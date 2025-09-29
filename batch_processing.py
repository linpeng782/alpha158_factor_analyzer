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
