"""
单独运行回测分析（使用已生成的预测结果）

使用方法：
    python run_backtest_only.py --predictions predictions_2024.csv --start 2024-01-01 --end 2024-12-31
"""

import argparse
import pandas as pd
import copy
from pathlib import Path

import qlib
from qlib.constant import REG_CN


def run_backtest_from_csv(predictions_csv, start_time, end_time, benchmark="SH000300"):
    """
    从CSV文件读取预测结果并运行回测

    Args:
        predictions_csv: 预测结果CSV文件路径
        start_time: 回测开始时间
        end_time: 回测结束时间
        benchmark: 基准指数
    """
    print("="*60)
    print("Running Backtest from Predictions")
    print("="*60)

    # 加载预测结果
    print(f"\nLoading predictions from: {predictions_csv}")
    pred_df = pd.read_csv(predictions_csv)

    # 转换为 Qlib 需要的格式
    pred_df['datetime'] = pd.to_datetime(pred_df['datetime'])
    pred = pred_df.set_index(['datetime', 'instrument'])['score']

    print(f"✓ Loaded {len(pred)} predictions")

    # 获取预测数据的实际日期范围
    pred_start = pred.index.get_level_values(0).min()
    pred_end = pred.index.get_level_values(0).max()
    print(f"  Date range: {pred_start} to {pred_end}")
    print(f"  Instruments: {pred.index.get_level_values(1).nunique()}")

    # 调整回测日期范围以匹配预测数据
    # 如果用户指定的日期范围超出预测数据范围，自动调整
    actual_start = max(pd.to_datetime(start_time), pred_start).strftime('%Y-%m-%d')
    # 回测结束日期设置为预测最后日期的前一天，避免索引越界
    actual_end_date = min(pd.to_datetime(end_time), pred_end)
    # 尝试往前推一天作为安全边界
    safe_end = (actual_end_date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    actual_end = safe_end

    if actual_start != start_time or actual_end != end_time:
        print(f"\n⚠️  Adjusting backtest period to match prediction data:")
        print(f"  Requested: {start_time} to {end_time}")
        print(f"  Adjusted:  {actual_start} to {actual_end}")
        print(f"  Note: End date adjusted to avoid calendar boundary issues")
        start_time = actual_start
        end_time = actual_end

    # 回测配置
    backtest_config = {
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy",
            "kwargs": {
                "signal": pred,
                "topk": 50,
                "n_drop": 5,
            },
        },
        "backtest": {
            "start_time": start_time,
            "end_time": end_time,
            "account": 100000000,
            "benchmark": benchmark,
            "exchange_kwargs": {
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }

    print(f"\n{'='*60}")
    print(f"Backtest Configuration")
    print(f"{'='*60}")
    print(f"  Strategy: TopkDropoutStrategy")
    print(f"  Period: {start_time} to {end_time}")
    print(f"  Benchmark: {benchmark}")
    print(f"  Initial account: 100,000,000")

    # 执行回测
    try:
        from qlib.backtest import backtest as normal_backtest
        from qlib.contrib.evaluate import risk_analysis

        print(f"\nExecuting backtest...")
        portfolio_metric_dict, indicator_dict = normal_backtest(
            executor=backtest_config.get("executor", {
                "class": "SimulatorExecutor",
                "module_path": "qlib.backtest.executor",
                "kwargs": {
                    "time_per_step": "day",
                    "generate_portfolio_metrics": True,
                },
            }),
            strategy=backtest_config["strategy"],
            **backtest_config["backtest"]
        )

        # 分析结果
        for freq, (report_normal, positions_normal) in portfolio_metric_dict.items():
            print(f"\n{'='*60}")
            print(f"Backtest Results (Frequency: {freq})")
            print(f"{'='*60}")

            # 计算各项指标
            bench_return = report_normal["bench"]
            strategy_return = report_normal["return"]
            cost = report_normal["cost"]

            # 基准收益分析
            print(f"\n【Benchmark Performance】")
            bench_analysis = risk_analysis(bench_return, freq=freq).to_dict()["risk"]
            print(f"  Annualized Return: {bench_analysis['annualized_return']:.4f} ({bench_analysis['annualized_return']*100:.2f}%)")
            print(f"  Information Ratio:  {bench_analysis['information_ratio']:.4f}")
            print(f"  Max Drawdown:       {bench_analysis['max_drawdown']:.4f} ({bench_analysis['max_drawdown']*100:.2f}%)")

            # 超额收益分析（无成本）
            print(f"\n【Excess Return (Without Cost)】")
            excess_return_no_cost = strategy_return - bench_return
            analysis_no_cost = risk_analysis(excess_return_no_cost, freq=freq).to_dict()["risk"]
            print(f"  Annualized Return: {analysis_no_cost['annualized_return']:.4f} ({analysis_no_cost['annualized_return']*100:.2f}%)")
            print(f"  Information Ratio:  {analysis_no_cost['information_ratio']:.4f}")
            print(f"  Max Drawdown:       {analysis_no_cost['max_drawdown']:.4f} ({analysis_no_cost['max_drawdown']*100:.2f}%)")

            # 超额收益分析（含成本）
            print(f"\n【Excess Return (With Cost)】")
            excess_return_with_cost = strategy_return - bench_return - cost
            analysis_with_cost = risk_analysis(excess_return_with_cost, freq=freq).to_dict()["risk"]
            print(f"  Annualized Return: {analysis_with_cost['annualized_return']:.4f} ({analysis_with_cost['annualized_return']*100:.2f}%)")
            print(f"  Information Ratio:  {analysis_with_cost['information_ratio']:.4f}")
            print(f"  Max Drawdown:       {analysis_with_cost['max_drawdown']:.4f} ({analysis_with_cost['max_drawdown']*100:.2f}%)")

            # 关键指标汇总
            print(f"\n【Key Metrics Summary】")
            print(f"  Strategy Annualized Return: {analysis_with_cost['annualized_return']:.4f} ({analysis_with_cost['annualized_return']*100:.2f}%)")
            print(f"  Strategy Sharpe Ratio:      {analysis_with_cost['information_ratio']:.4f}")
            print(f"  Strategy Max Drawdown:      {analysis_with_cost['max_drawdown']:.4f} ({analysis_with_cost['max_drawdown']*100:.2f}%)")

            # 计算总交易成本
            total_cost = cost.sum()
            print(f"\n【Trading Costs】")
            print(f"  Total Cost: {total_cost:.6f}")
            print(f"  Daily Avg:  {cost.mean():.6f}")

        print(f"\n✓ Backtest completed successfully!")

    except Exception as e:
        print(f"\n⚠ Backtest failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Run backtest analysis using pre-generated predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--predictions',
        type=str,
        default='predictions_2024.csv',
        help='Path to predictions CSV file (default: predictions_2024.csv)'
    )
    parser.add_argument(
        '--start',
        type=str,
        default='2024-01-01',
        help='Backtest start date (default: 2024-01-01)'
    )
    parser.add_argument(
        '--end',
        type=str,
        default='2024-12-31',
        help='Backtest end date (default: 2024-12-31)'
    )
    parser.add_argument(
        '--benchmark',
        type=str,
        default='SH000300',
        help='Benchmark index (default: SH000300)'
    )

    args = parser.parse_args()

    # 检查文件是否存在
    if not Path(args.predictions).exists():
        print(f"❌ Error: Predictions file not found: {args.predictions}")
        print(f"\nAvailable prediction files:")
        for f in Path('.').glob('predictions_*.csv'):
            print(f"  - {f}")
        return

    # 初始化 Qlib
    print("Initializing Qlib...")
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    print("✓ Qlib initialized")

    # 运行回测
    run_backtest_from_csv(
        predictions_csv=args.predictions,
        start_time=args.start,
        end_time=args.end,
        benchmark=args.benchmark
    )


if __name__ == "__main__":
    main()
