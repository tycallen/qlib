"""
加载训练好的模型权重，复现回测结果并推理新数据

功能：
1. 加载已训练的模型权重
2. 复现训练时的回测结果
3. 对新的时间段数据进行推理
4. 生成分析报告
"""

import pickle
import torch
import pandas as pd
import numpy as np
from pathlib import Path

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


def load_trained_model(model_pkl_path):
    """
    加载训练好的Qlib模型对象

    Args:
        model_pkl_path: 模型pkl文件路径（从mlruns中获取）

    Returns:
        加载的模型对象
    """
    print(f"正在加载模型: {model_pkl_path}")

    with open(model_pkl_path, 'rb') as f:
        model = pickle.load(f)

    print(f"✓ 模型加载成功")
    print(f"  模型类型: {type(model).__name__}")
    print(f"  模型类: {model.__class__.__module__}.{model.__class__.__name__}")

    # 检查模型是否已经训练
    if hasattr(model, 'fitted') and model.fitted:
        print(f"  状态: 已训练")
    else:
        print(f"  警告: 模型可能未训练")

    return model


def reproduce_backtest(model, dataset_config, backtest_config):
    """
    使用加载的模型复现回测结果

    Args:
        model: 已加载的模型
        dataset_config: 数据集配置
        backtest_config: 回测配置
    """
    print("\n" + "="*60)
    print("复现回测结果")
    print("="*60)

    # 创建数据集
    dataset = init_instance_by_config(dataset_config)

    # 使用模型进行预测
    print("\n正在预测...")
    pred = model.predict(dataset, segment="test")
    print(f"✓ 预测完成，共 {len(pred)} 条数据")

    # 显示预测结果统计
    print(f"\n预测结果统计:")
    print(f"  均值: {pred.mean():.6f}")
    print(f"  标准差: {pred.std():.6f}")
    print(f"  最小值: {pred.min():.6f}")
    print(f"  最大值: {pred.max():.6f}")
    print(f"  NaN数量: {pred.isna().sum()}")

    # 进行回测
    print("\n正在进行回测分析...")

    # 更新回测配置中的信号
    backtest_config["strategy"]["kwargs"]["signal"] = pred

    # 创建回测记录
    with R.start(experiment_name="model_reproduce"):
        # 保存预测结果
        R.save_objects(**{"pred.pkl": pred})

        # 信号分析
        print("\n生成信号分析...")
        sig_rec = SigAnaRecord(recorder=R)
        sig_rec.generate(pred=pred, dataset=dataset)

        # 组合分析（回测）
        print("\n生成组合分析（回测）...")
        port_rec = PortAnaRecord(recorder=R, config=backtest_config)
        port_rec.generate()

        print("\n✓ 回测分析完成")
        print(f"✓ 实验ID: {R.get_exp_id()}")
        print(f"✓ 记录URI: {R.get_uri()}")

    return pred


def predict_new_data(model, dataset_config, new_test_segment):
    """
    使用模型对新的时间段数据进行推理

    Args:
        model: 已加载的模型
        dataset_config: 数据集配置（会修改test时间段）
        new_test_segment: 新的测试时间段，格式如 ("2021-01-01", "2022-12-31")

    Returns:
        新数据的预测结果
    """
    print("\n" + "="*60)
    print("对新数据进行推理")
    print("="*60)
    print(f"新的测试时间段: {new_test_segment[0]} 至 {new_test_segment[1]}")

    # 更新数据集配置中的test时间段
    new_dataset_config = dataset_config.copy()
    new_dataset_config["kwargs"]["segments"]["test"] = new_test_segment

    # 创建新数据集
    dataset = init_instance_by_config(new_dataset_config)

    # 预测
    print("\n正在预测...")
    pred = model.predict(dataset, segment="test")
    print(f"✓ 预测完成，共 {len(pred)} 条数据")

    # 显示预测结果统计
    print(f"\n预测结果统计:")
    print(f"  均值: {pred.mean():.6f}")
    print(f"  标准差: {pred.std():.6f}")
    print(f"  最小值: {pred.min():.6f}")
    print(f"  最大值: {pred.max():.6f}")
    print(f"  NaN数量: {pred.isna().sum()}")

    # 显示前10条预测结果
    print(f"\n前10条预测结果:")
    print(pred.head(10))

    return pred


def analyze_predictions(pred, dataset, segment="test"):
    """
    分析预测结果

    Args:
        pred: 预测结果
        dataset: 数据集
        segment: 数据段
    """
    print("\n" + "="*60)
    print("预测分析")
    print("="*60)

    # 获取标签数据
    df_test = dataset.prepare(segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    label = df_test["label"]

    # 对齐预测和标签的索引
    common_idx = pred.index.intersection(label.index)
    pred_aligned = pred.loc[common_idx]
    label_aligned = label.loc[common_idx]

    # 计算相关性
    mask = (~pred_aligned.isna()) & (~label_aligned.isna())
    if mask.sum() > 0:
        correlation = pred_aligned[mask].corr(label_aligned[mask].squeeze())
        print(f"预测与标签的相关性: {correlation:.6f}")
    else:
        print("无法计算相关性（数据不足）")

    # 按日期统计
    daily_stats = pred.groupby(level=0).agg(['mean', 'std', 'count'])
    print(f"\n每日预测统计（前5天）:")
    print(daily_stats.head())

    return daily_stats


def save_predictions(pred, output_path="predictions.csv"):
    """
    保存预测结果到文件

    Args:
        pred: 预测结果
        output_path: 输出文件路径
    """
    print(f"\n保存预测结果到: {output_path}")

    # 转换为DataFrame并保存
    pred_df = pred.reset_index()
    pred_df.columns = ['datetime', 'instrument', 'score']
    pred_df.to_csv(output_path, index=False)

    print(f"✓ 预测结果已保存")
    print(f"  文件大小: {Path(output_path).stat().st_size / 1024:.2f} KB")


def main():
    """主函数"""

    # ============================================================
    # 配置部分 - 根据你的实际情况修改
    # ============================================================

    # 1. 模型路径（从mlruns中找到训练好的模型）
    MODEL_PKL_PATH = "./mlruns/686249546704553370/2ee5b9c2d8f2420e8cb39aa84b35546e/artifacts/params.pkl"

    # 2. 初始化Qlib
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

    # 3. 数据集配置（与训练时一致）
    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha360",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": "2008-01-01",
                    "end_time": "2020-08-01",
                    "fit_start_time": "2008-01-01",
                    "fit_end_time": "2014-12-31",
                    "instruments": "csi300",
                },
            },
            "segments": {
                "train": ("2008-01-01", "2014-12-31"),
                "valid": ("2015-01-01", "2016-12-31"),
                "test": ("2017-01-01", "2020-08-01"),
            },
        },
    }

    # 4. 回测配置
    backtest_config = {
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy",
            "kwargs": {
                "signal": None,  # 会在运行时填充
                "topk": 50,
                "n_drop": 5,
            },
        },
        "backtest": {
            "start_time": "2017-01-01",
            "end_time": "2020-08-01",
            "account": 100000000,
            "benchmark": "SH000300",
            "exchange_kwargs": {
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }

    # ============================================================
    # 执行部分
    # ============================================================

    print("="*60)
    print("模型加载与预测")
    print("="*60)

    # 步骤1: 加载模型
    model = load_trained_model(MODEL_PKL_PATH)

    # 步骤2: 复现原始回测结果
    print("\n" + "="*60)
    print("步骤1: 复现原始测试集的回测结果")
    print("="*60)
    pred_original = reproduce_backtest(model, dataset_config, backtest_config)

    # 保存原始预测结果
    save_predictions(pred_original, "predictions_original_test.csv")

    # 步骤3: 对新数据进行推理（例如2020年下半年到2021年）
    print("\n" + "="*60)
    print("步骤2: 对新时间段数据进行推理")
    print("="*60)

    # 修改为你想要预测的新时间段
    NEW_TEST_SEGMENT = ("2020-08-01", "2021-12-31")

    try:
        pred_new = predict_new_data(model, dataset_config, NEW_TEST_SEGMENT)
        save_predictions(pred_new, "predictions_new_data.csv")

        # 如果需要，也可以对新数据进行回测
        print("\n对新数据进行回测...")
        new_backtest_config = backtest_config.copy()
        new_backtest_config["backtest"]["start_time"] = NEW_TEST_SEGMENT[0]
        new_backtest_config["backtest"]["end_time"] = NEW_TEST_SEGMENT[1]
        new_backtest_config["strategy"]["kwargs"]["signal"] = pred_new

        with R.start(experiment_name="model_new_data"):
            R.save_objects(**{"pred.pkl": pred_new})
            port_rec = PortAnaRecord(recorder=R, config=new_backtest_config)
            port_rec.generate()
            print(f"\n✓ 新数据回测完成")
            print(f"✓ 实验ID: {R.get_exp_id()}")

    except Exception as e:
        print(f"\n警告: 新数据推理失败: {e}")
        print("这可能是因为数据不存在或数据范围超出可用数据")

    # 步骤4: 使用简化方式快速推理
    print("\n" + "="*60)
    print("步骤3: 快速推理示例")
    print("="*60)
    print("\n使用代码快速推理:")
    print("""
# 简化版本 - 只需要加载模型和数据集
import pickle
from qlib.data.dataset import DatasetH
from qlib.utils import init_instance_by_config

# 加载模型
with open('params.pkl', 'rb') as f:
    model = pickle.load(f)

# 创建数据集（使用你的配置）
dataset = init_instance_by_config(dataset_config)

# 预测
pred = model.predict(dataset, segment="test")

# 保存结果
pred.to_csv("my_predictions.csv")
    """)

    print("\n" + "="*60)
    print("全部完成！")
    print("="*60)
    print("\n生成的文件:")
    print("  - predictions_original_test.csv: 原始测试集预测结果")
    print("  - predictions_new_data.csv: 新数据预测结果")
    print("\n查看实验结果:")
    print("  使用 qlib_mlflow 命令查看MLflow UI")


if __name__ == "__main__":
    main()
