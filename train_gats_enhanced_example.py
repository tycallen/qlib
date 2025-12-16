"""
Enhanced GAT_ts Training Example

Features:
1. Best checkpoint automatic saving
2. Graceful Ctrl+C handling
3. Enhanced training logs
4. MLflow experiment tracking
5. Resume from checkpoint
6. Support loading config from YAML file

Usage:
    python train_gats_enhanced_example.py --config examples/benchmarks/GATs/workflow_config_gats_Alpha158.yaml
    python train_gats_enhanced_example.py --config examples/benchmarks/GATs/workflow_config_gats_Alpha158.yaml --resume
"""

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
# from qlib.contrib.model.pytorch_gats_ts_enhanced import GATs
from qlib.contrib.model.pytorch_gats_ts_optimized import GATs
import mlflow
import yaml
import argparse
from pathlib import Path


def load_config(config_path):
    """加载YAML配置文件"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print(f"✓ 已加载配置文件: {config_path}")
    return config


def train_gats_with_enhancements(config=None):
    """Train GAT model with all enhancements"""

    # ============================================================
    # 1. Initialize Qlib
    # ============================================================
    print("="*60)
    print("Enhanced GAT Training")
    print("="*60)

    # 从配置文件读取Qlib初始化参数
    if config:
        qlib_config = config.get('qlib_init', {})
        provider_uri = qlib_config.get('provider_uri', '~/.qlib/qlib_data/cn_data')
        region = qlib_config.get('region', 'cn')
    else:
        provider_uri = "~/.qlib/qlib_data/cn_data"
        region = "cn"

    qlib.init(provider_uri=provider_uri, region=region)
    print(f"✓ Qlib initialized: {provider_uri}")

    # ============================================================
    # 2. Create Dataset
    # ============================================================
    if config and 'task' in config:
        # 从配置文件读取数据集配置
        dataset_config = config['task']['dataset']
        print(f"✓ Using dataset config from YAML")
    else:
        # 默认配置（向后兼容）
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

    dataset = init_instance_by_config(dataset_config)
    print(f"✓ Dataset created")

    # ============================================================
    # 3. Create Enhanced GAT Model
    # ============================================================
    if config and 'task' in config:
        # 从配置文件读取模型配置
        model_config = config['task']['model']['kwargs']
        print(f"✓ Using model config from YAML")
    else:
        # 默认配置（向后兼容）
        model_config = {
            "d_feat": 6,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.0,
            "n_epochs": 200,
            "lr": 0.001,
            "metric": "loss",
            "early_stop": 20,
            "loss": "mse",
            "base_model": "LSTM",
            "model_path": None,
            "optimizer": "adam",
            "GPU": 0,
            "n_jobs": 10,
            "seed": 42,
            "checkpoint_dir": "./checkpoints/gats_enhanced",
            "save_checkpoint_interval": 5,
            "use_mlflow": True,
        }

    model = GATs(**model_config)
    print(f"✓ Model created: {model_config.get('base_model', 'LSTM')} based GAT")

    # ============================================================
    # 4. Train with MLflow Tracking
    # ============================================================
    # 从配置文件获取实验名称和保存路径
    if config:
        experiment_name = config.get('experiment', {}).get('name', 'gats_training')
        save_path = config.get('experiment', {}).get('save_path', 'gats_model.pth')
    else:
        experiment_name = "gats_enhanced_training"
        save_path = "gats_enhanced_model.pth"

    print("\n开始训练...")
    print("提示: 按 Ctrl+C 可以随时中断训练，会自动保存最佳checkpoint")
    print("="*60 + "\n")

    with R.start(experiment_name=experiment_name):
        # Log additional info
        R.log_params(
            model_type="GATs_Optimized",
            dataset=dataset_config.get('kwargs', {}).get('handler', {}).get('class', 'Unknown'),
            base_model=model_config.get('base_model', 'LSTM'),
        )

        try:
            # Train the model
            model.fit(dataset, save_path=save_path)

            print("\n训练完成！")

            # ============================================================
            # 5. Predict and Backtest
            # ============================================================
            print("\n开始预测和回测...")
            pred = model.predict(dataset)

            # Save predictions
            R.save_objects(**{"pred.pkl": pred})

            # Save model weights to MLflow artifacts
            import os
            if os.path.exists(save_path):
                R.save_objects(local_path=save_path)
                print(f"✓ Model weights saved to MLflow: {save_path}")

            # Optionally save checkpoints directory if it exists
            checkpoint_dir = model_config.get('checkpoint_dir', './checkpoints')
            if os.path.exists(checkpoint_dir):
                # Save only the best checkpoint to reduce storage
                best_checkpoint = os.path.join(checkpoint_dir, "best_checkpoint.pth")
                if os.path.exists(best_checkpoint):
                    R.save_objects(local_path=best_checkpoint, artifact_path="checkpoints")
                    print(f"✓ Best checkpoint saved to MLflow: {best_checkpoint}")

            # Get recorder object
            recorder = R.get_recorder()

            # Signal analysis
            sig_rec = SignalRecord(model=model, dataset=dataset, recorder=recorder)
            sig_rec.generate()

            # Backtest
            if config and 'port_analysis_config' in config:
                # 从配置文件读取回测配置
                backtest_config = config['port_analysis_config']
                # 更新signal
                backtest_config['strategy']['kwargs']['signal'] = pred
                print(f"✓ Using backtest config from YAML")
            else:
                # 默认配置
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

            port_rec = PortAnaRecord(recorder=recorder, config=backtest_config)
            port_rec.generate()

            print("\n✓ 全部完成！")
            print(f"实验ID: {recorder.experiment_id}")
            print(f"记录ID: {recorder.id}")

        except KeyboardInterrupt:
            print("\n检测到 Ctrl+C，正在保存最佳checkpoint...")
            print("最佳checkpoint已保存到: ./checkpoints/gats_enhanced/best_checkpoint.pth")
            print("\n可以使用以下代码加载最佳模型继续回测:")
            print("""
from qlib.contrib.model.pytorch_gats_ts_enhanced import GATs
import torch

# 加载模型
model = GATs(...)
checkpoint = torch.load("./checkpoints/gats_enhanced/best_checkpoint.pth")
model.GAT_model.load_state_dict(checkpoint['model_state_dict'])
model.fitted = True

# 继续回测
pred = model.predict(dataset)
# ... 进行回测
            """)

        print("\n查看训练结果:")
        print("  1. 查看checkpoints: ls -lh ./checkpoints/gats_enhanced/")
        print("  2. 启动MLflow UI: mlflow ui")
        print("  3. 浏览器访问: http://localhost:5000")


def resume_from_checkpoint(config=None):
    """从checkpoint恢复并继续回测"""

    print("="*60)
    print("从Checkpoint恢复")
    print("="*60)

    # Initialize Qlib
    if config:
        qlib_config = config.get('qlib_init', {})
        provider_uri = qlib_config.get('provider_uri', '~/.qlib/qlib_data/cn_data')
        region = qlib_config.get('region', 'cn')
    else:
        provider_uri = "~/.qlib/qlib_data/cn_data"
        region = "cn"

    qlib.init(provider_uri=provider_uri, region=region)

    # Create dataset
    if config and 'task' in config:
        # 从配置文件读取，但只需要test段
        dataset_config = config['task']['dataset']
        print(f"✓ Using dataset config from YAML")
    else:
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
                        "instruments": "csi300",
                    },
                },
                "segments": {
                    "test": ("2017-01-01", "2020-08-01"),
                },
            },
        }
    dataset = init_instance_by_config(dataset_config)

    # Create model
    if config and 'task' in config:
        model_config = config['task']['model']['kwargs']
        # 只需要关键参数来创建模型结构
        model = GATs(
            d_feat=model_config.get('d_feat', 6),
            hidden_size=model_config.get('hidden_size', 64),
            num_layers=model_config.get('num_layers', 2),
            dropout=model_config.get('dropout', 0.0),
            base_model=model_config.get('base_model', 'LSTM'),
            GPU=model_config.get('GPU', 0),  # 添加GPU参数
            gpus=model_config.get('gpus', None),  # 添加gpus参数（多GPU）
        )
        checkpoint_dir = model_config.get('checkpoint_dir', './checkpoints')
    else:
        model = GATs(
            d_feat=6,
            hidden_size=64,
            num_layers=2,
            dropout=0.0,
            base_model="LSTM",
            GPU=0,
        )
        checkpoint_dir = "./checkpoints"

    # Load best checkpoint
    checkpoint_path = f"{checkpoint_dir}/best_checkpoint.pth"
    print(f"\n加载checkpoint: {checkpoint_path}")

    checkpoint = model.load_checkpoint(checkpoint_path)
    model.fitted = True

    print(f"✓ 已恢复到 Epoch {checkpoint['epoch']+1}")
    print(f"  Score: {checkpoint['score']:.6f}")
    print(f"  Best Score: {checkpoint['best_score']:.6f} @ Epoch {checkpoint['best_epoch']+1}")

    # Predict
    print("\n开始预测...")
    pred = model.predict(dataset)
    print(f"✓ 预测完成: {len(pred)} 条数据")

    # Backtest
    print("\n开始回测...")
    with R.start(experiment_name="gats_resumed_from_checkpoint"):
        R.log_params({
            "resumed_from": checkpoint_path,
            "resumed_epoch": checkpoint['epoch'],
            "resumed_score": checkpoint['score'],
        })

        R.save_objects(**{"pred.pkl": pred})

        # Get recorder object
        recorder = R.get_recorder()

        # 回测配置
        if config and 'port_analysis_config' in config:
            backtest_config = config['port_analysis_config']
            backtest_config['strategy']['kwargs']['signal'] = pred
        else:
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

        port_rec = PortAnaRecord(recorder=recorder, config=backtest_config)
        port_rec.generate()

        print("\n✓ 回测完成！")
        print(f"实验ID: {recorder.experiment_id}")
        print(f"记录ID: {recorder.id}")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Enhanced GAT Training with Config File Support')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (e.g., examples/benchmarks/GATs/workflow_config_gats_Alpha158.yaml)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint and run backtest only')

    args = parser.parse_args()

    # 加载配置文件（如果指定）
    config = None
    if args.config:
        try:
            config = load_config(args.config)
        except Exception as e:
            print(f"错误: 无法加载配置文件: {e}")
            exit(1)

    # 执行训练或恢复
    if args.resume:
        # Resume from checkpoint
        resume_from_checkpoint(config)
    else:
        # Normal training
        train_gats_with_enhancements(config)
