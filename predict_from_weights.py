"""
直接从 .pth 权重文件加载模型进行推理和回测

功能：
1. 从 gats_model.pth 权重文件加载模型
2. 对指定时间段数据进行推理
3. 执行回测分析
4. 保存预测结果为 CSV 文件

使用方法：
    python predict_from_weights.py
"""

import torch
import pandas as pd
import numpy as np
import copy
from pathlib import Path

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.model.pytorch_gats_ts_optimized import GATs
import yaml


def load_config_from_yaml(config_path):
    """从 YAML 文件加载配置"""
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"✓ Config loaded successfully")
    return config


def load_model_from_weights(weights_path, model_config):
    """
    从 .pth 权重文件加载模型

    Args:
        weights_path: 权重文件路径 (例如 gats_model.pth)
        model_config: 模型配置字典

    Returns:
        加载好权重的模型对象
    """
    print(f"\nLoading model from weights: {weights_path}")

    # 创建模型实例
    # GPU 配置：-1 = CPU, 0/1/2 = 单GPU, [0,1,2] = 多GPU
    use_gpu = model_config.get('GPU', -1)
    use_gpus = model_config.get('gpus', None)

    print(f"\nGPU Configuration:")
    if use_gpus is not None:
        print(f"  Mode: Multi-GPU")
        print(f"  GPUs: {use_gpus}")
    elif use_gpu >= 0:
        print(f"  Mode: Single GPU")
        print(f"  GPU: {use_gpu}")
    else:
        print(f"  Mode: CPU")

    # 注意：不在初始化时传入 gpus 参数，避免模型被提前用 DataParallel 包装
    # 我们将在加载权重后手动包装
    model = GATs(
        d_feat=model_config.get('d_feat', 6),
        hidden_size=model_config.get('hidden_size', 64),
        num_layers=model_config.get('num_layers', 2),
        dropout=model_config.get('dropout', 0.0),
        n_epochs=model_config.get('n_epochs', 200),
        lr=model_config.get('lr', 0.001),
        metric=model_config.get('metric', 'loss'),
        early_stop=model_config.get('early_stop', 20),
        loss=model_config.get('loss', 'mse'),
        base_model=model_config.get('base_model', 'LSTM'),
        optimizer=model_config.get('optimizer', 'adam'),
        GPU=use_gpu if use_gpus is None else -1,  # 多GPU时先用CPU初始化，避免提前包装
        gpus=None,  # 不在这里传入，手动包装
        n_jobs=model_config.get('n_jobs', 10),
        seed=model_config.get('seed', 42),
    )

    print(f"✓ Model architecture created")
    print(f"  Base model: {model_config.get('base_model', 'LSTM')}")
    print(f"  Hidden size: {model_config.get('hidden_size', 64)}")
    print(f"  Num layers: {model_config.get('num_layers', 2)}")

    # 加载权重
    if Path(weights_path).exists():
        print(f"\nLoading weights...")

        # 根据配置决定加载到哪个设备
        if use_gpus is not None or use_gpu >= 0:
            # GPU模式
            device = torch.device(f"cuda:{use_gpu if use_gpu >= 0 else 0}")
            state_dict = torch.load(weights_path, map_location=device)
        else:
            # CPU模式
            device = torch.device('cpu')
            state_dict = torch.load(weights_path, map_location='cpu')

        # 处理可能的 DataParallel 包装的权重
        # 训练时保存的 state_dict 可能带或不带 'module.' 前缀
        if list(state_dict.keys())[0].startswith('module.'):
            # 移除 'module.' 前缀（因为我们先加载到原始模型）
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # 移除 'module.'
                new_state_dict[name] = v
            state_dict = new_state_dict
            print(f"  Removed 'module.' prefix from state_dict keys")

        # 先加载权重到原始模型（未包装的）
        model.GAT_model.load_state_dict(state_dict)
        model.fitted = True
        print(f"✓ Weights loaded successfully")

        # 移动模型到指定设备
        model.GAT_model.to(device)
        model.device = device  # 更新模型的 device 属性
        print(f"✓ Model moved to: {device}")

        # 如果使用多GPU，再包装模型（在加载权重之后）
        if use_gpus is not None:
            import torch.nn as nn
            model.GAT_model = nn.DataParallel(model.GAT_model, device_ids=use_gpus)
            # 更新 device 属性，确保推理时正确
            model.device = device  # DataParallel的主设备
            print(f"✓ Model wrapped with DataParallel on GPUs: {use_gpus}")

        # 设置为评估模式
        model.GAT_model.eval()
        print(f"✓ Model set to evaluation mode")
    else:
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    return model


def predict_and_save(model, dataset, segment="test", output_path="predictions.csv"):
    """
    使用模型进行预测并保存结果

    Args:
        model: 已加载权重的模型
        dataset: 数据集对象
        segment: 数据段名称 (train/valid/test)
        output_path: 输出CSV文件路径

    Returns:
        预测结果 DataFrame
    """
    print(f"\n{'='*60}")
    print(f"Predicting on {segment} set")
    print(f"{'='*60}")

    # 预测
    print(f"Running prediction...")
    pred = model.predict(dataset)

    print(f"✓ Prediction completed")
    print(f"  Total samples: {len(pred)}")

    # 显示预测统计
    print(f"\nPrediction Statistics:")
    print(f"  Mean:   {pred.mean():.6f}")
    print(f"  Std:    {pred.std():.6f}")
    print(f"  Min:    {pred.min():.6f}")
    print(f"  Max:    {pred.max():.6f}")
    print(f"  NaN:    {pred.isna().sum()}")

    # 显示前几条预测结果
    print(f"\nFirst 10 predictions:")
    print(pred.head(10))

    # 保存为 CSV
    print(f"\nSaving predictions to: {output_path}")
    pred_df = pred.reset_index()
    pred_df.columns = ['datetime', 'instrument', 'score']
    pred_df.to_csv(output_path, index=False)

    file_size = Path(output_path).stat().st_size / 1024
    print(f"✓ Predictions saved")
    print(f"  File size: {file_size:.2f} KB")
    print(f"  Rows: {len(pred_df)}")

    return pred


def run_backtest(pred, backtest_config):
    """
    执行回测分析

    Args:
        pred: 预测结果
        backtest_config: 回测配置
    """
    print(f"\n{'='*60}")
    print(f"Running Backtest Analysis")
    print(f"{'='*60}")

    try:
        from qlib.backtest import backtest as normal_backtest
        from qlib.contrib.evaluate import risk_analysis

        # 更新回测配置
        backtest_config_copy = copy.deepcopy(backtest_config)
        backtest_config_copy["strategy"]["kwargs"]["signal"] = pred

        print(f"\nBacktest configuration:")
        print(f"  Strategy: {backtest_config_copy['strategy']['class']}")
        print(f"  Period: {backtest_config_copy['backtest']['start_time']} to {backtest_config_copy['backtest']['end_time']}")
        print(f"  Benchmark: {backtest_config_copy['backtest']['benchmark']}")
        print(f"  Initial account: {backtest_config_copy['backtest']['account']:,.0f}")

        # 执行回测
        print(f"\nExecuting backtest...")
        portfolio_metric_dict, indicator_dict = normal_backtest(
            executor=backtest_config_copy.get("executor", {
                "class": "SimulatorExecutor",
                "module_path": "qlib.backtest.executor",
                "kwargs": {
                    "time_per_step": "day",
                    "generate_portfolio_metrics": True,
                },
            }),
            strategy=backtest_config_copy["strategy"],
            **backtest_config_copy["backtest"]
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

        print(f"\n✓ Backtest analysis completed")

    except Exception as e:
        print(f"\n⚠ Backtest analysis failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""

    print("="*60)
    print("Model Prediction from Weights")
    print("="*60)

    # ============================================================
    # 配置部分 - 根据实际情况修改
    # ============================================================

    # 1. 路径配置（自动检测 macOS 或 Linux）
    import os
    if os.path.exists("/Volumes/data10"):
        # macOS with mounted volume
        BASE_PATH = "/Volumes/data10/tyc/quant/qlib_own"
    else:
        # Linux
        BASE_PATH = "/data10/tyc/quant/qlib_own"

    WEIGHTS_PATH = f"{BASE_PATH}/gats_model.pth"
    CONFIG_PATH = f"{BASE_PATH}/examples/benchmarks/GATs/workflow_config_gats_Alpha158.yaml"

    # 2. 预测时间段配置
    # 原始测试集
    ORIGINAL_TEST_SEGMENT = ("2024-01-01", "2024-12-31")
    ORIGINAL_OUTPUT = "predictions_2024.csv"

    # 新数据推理
    NEW_TEST_SEGMENT = ("2025-08-09", "2025-11-21")
    NEW_OUTPUT = "predictions_2025.csv"

    # 3. 推理模式配置（重要！）
    # 选项3: 多GPU模式（推荐，与训练时一致）
    INFERENCE_MODE = "multi_gpu"
    INFERENCE_GPU = 0
    INFERENCE_GPUS = [0, 1, 2]  # 使用3张显卡

    # 选项1: CPU模式（最安全，但较慢）
    # INFERENCE_MODE = "cpu"
    # INFERENCE_GPU = -1
    # INFERENCE_GPUS = None

    # 选项2: 单GPU模式（如果单张显卡内存足够）
    # INFERENCE_MODE = "single_gpu"
    # INFERENCE_GPU = 0
    # INFERENCE_GPUS = None

    # 选项3: 多GPU模式（推荐，与训练时一致）
    # INFERENCE_MODE = "multi_gpu"
    # INFERENCE_GPU = 0
    # INFERENCE_GPUS = [0, 1, 2]  # 使用3张显卡

    print(f"\n{'='*60}")
    print(f"Inference Configuration")
    print(f"{'='*60}")
    print(f"  Mode: {INFERENCE_MODE}")
    if INFERENCE_MODE == "cpu":
        print(f"  Device: CPU")
    elif INFERENCE_MODE == "single_gpu":
        print(f"  Device: GPU {INFERENCE_GPU}")
    elif INFERENCE_MODE == "multi_gpu":
        print(f"  Devices: GPUs {INFERENCE_GPUS}")

    # 4. 是否执行回测（可能较慢）
    RUN_BACKTEST = True

    # ============================================================
    # 执行部分
    # ============================================================

    # 步骤1: 初始化 Qlib
    print("\nStep 1: Initializing Qlib...")
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    print("✓ Qlib initialized")

    # 步骤2: 加载配置
    print("\nStep 2: Loading configuration...")
    config = load_config_from_yaml(CONFIG_PATH)

    dataset_config = config['task']['dataset']
    model_config = config['task']['model']['kwargs']
    backtest_config = config.get('port_analysis_config', None)

    # 覆盖GPU配置为推理模式
    model_config['GPU'] = INFERENCE_GPU
    model_config['gpus'] = INFERENCE_GPUS

    print(f"✓ Configuration loaded")
    print(f"  Handler: {dataset_config['kwargs']['handler']['class']}")
    print(f"  Base Model: {model_config['base_model']}")
    print(f"✓ GPU config overridden for inference")

    # 步骤3: 加载模型权重
    print("\nStep 3: Loading model weights...")
    model = load_model_from_weights(WEIGHTS_PATH, model_config)

    # 步骤4: 对原始测试集进行预测
    print("\n" + "="*60)
    print("Step 4: Predicting on Original Test Set")
    print("="*60)

    # 更新数据集配置的测试时间段
    dataset_config_original = copy.deepcopy(dataset_config)
    dataset_config_original['kwargs']['segments']['test'] = ORIGINAL_TEST_SEGMENT
    dataset_config_original['kwargs']['handler']['kwargs']['end_time'] = ORIGINAL_TEST_SEGMENT[1]

    # 创建数据集
    print(f"\nCreating dataset for {ORIGINAL_TEST_SEGMENT[0]} to {ORIGINAL_TEST_SEGMENT[1]}...")
    dataset_original = init_instance_by_config(dataset_config_original)
    print(f"✓ Dataset created")

    # 预测并保存
    pred_original = predict_and_save(
        model,
        dataset_original,
        segment="test",
        output_path=ORIGINAL_OUTPUT
    )

    # 回测
    if RUN_BACKTEST and backtest_config is not None:
        backtest_config_original = copy.deepcopy(backtest_config)
        backtest_config_original['backtest']['start_time'] = ORIGINAL_TEST_SEGMENT[0]
        backtest_config_original['backtest']['end_time'] = ORIGINAL_TEST_SEGMENT[1]
        run_backtest(pred_original, backtest_config_original)

    # 步骤5: 对新数据进行预测
    print("\n" + "="*60)
    print("Step 5: Predicting on New Data")
    print("="*60)

    try:
        # 更新数据集配置
        dataset_config_new = copy.deepcopy(dataset_config)
        dataset_config_new['kwargs']['segments']['test'] = NEW_TEST_SEGMENT
        dataset_config_new['kwargs']['handler']['kwargs']['end_time'] = NEW_TEST_SEGMENT[1]

        # 创建数据集
        print(f"\nCreating dataset for {NEW_TEST_SEGMENT[0]} to {NEW_TEST_SEGMENT[1]}...")
        dataset_new = init_instance_by_config(dataset_config_new)
        print(f"✓ Dataset created")

        # 预测并保存
        pred_new = predict_and_save(
            model,
            dataset_new,
            segment="test",
            output_path=NEW_OUTPUT
        )

        # 回测
        if RUN_BACKTEST and backtest_config is not None:
            backtest_config_new = copy.deepcopy(backtest_config)
            backtest_config_new['backtest']['start_time'] = NEW_TEST_SEGMENT[0]
            backtest_config_new['backtest']['end_time'] = NEW_TEST_SEGMENT[1]
            run_backtest(pred_new, backtest_config_new)

    except Exception as e:
        print(f"\n⚠ Warning: New data prediction failed: {e}")
        print(f"  This may be because data is not available for the specified period")
        import traceback
        traceback.print_exc()

    # 完成
    print("\n" + "="*60)
    print("All Tasks Completed!")
    print("="*60)
    print("\nGenerated files:")
    if Path(ORIGINAL_OUTPUT).exists():
        print(f"  ✓ {ORIGINAL_OUTPUT}")
    if Path(NEW_OUTPUT).exists():
        print(f"  ✓ {NEW_OUTPUT}")

    print("\nUsage examples:")
    print("  # View predictions")
    print(f"  head {ORIGINAL_OUTPUT}")
    print("  # Load in Python")
    print("  import pandas as pd")
    print(f"  pred = pd.read_csv('{ORIGINAL_OUTPUT}')")


if __name__ == "__main__":
    main()
