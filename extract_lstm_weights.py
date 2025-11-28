"""
从Qlib训练的LSTM模型中提取权重用于GAT初始化
"""

import pickle
import torch
from pathlib import Path


def extract_weights_from_qlib_lstm(model_pkl_path, output_path=None):
    """
    从Qlib保存的LSTM模型对象中提取PyTorch权重

    Args:
        model_pkl_path: Qlib保存的LSTM模型pkl文件路径
        output_path: 输出权重文件路径（.pth格式）
    """
    print(f"正在加载LSTM模型: {model_pkl_path}")

    # 加载Qlib模型对象
    with open(model_pkl_path, 'rb') as f:
        lstm_model_obj = pickle.load(f)

    print(f"模型类型: {type(lstm_model_obj)}")
    print(f"模型类名: {lstm_model_obj.__class__.__name__}")

    # 从Qlib的LSTM对象中提取PyTorch模型的state_dict
    if hasattr(lstm_model_obj, 'LSTM_model'):
        # pytorch_lstm_ts.py 中的LSTM类使用 LSTM_model 属性（大写）
        pytorch_model = lstm_model_obj.LSTM_model
        print(f"PyTorch模型类型: {type(pytorch_model)}")
    elif hasattr(lstm_model_obj, 'lstm_model'):
        # pytorch_lstm.py 中的LSTM类使用 lstm_model 属性（小写）
        pytorch_model = lstm_model_obj.lstm_model
        print(f"PyTorch模型类型: {type(pytorch_model)}")
    else:
        print("错误: 无法找到lstm_model或LSTM_model属性")
        print(f"可用属性: {[attr for attr in dir(lstm_model_obj) if not attr.startswith('_')]}")
        return None

    # 获取state_dict
    state_dict = pytorch_model.state_dict()

    print(f"\n模型权重层:")
    print("="*60)
    total_params = 0
    for key, value in state_dict.items():
        params = value.numel()
        total_params += params
        print(f"  {key:30s} {str(value.shape):20s} {params:>10,} 参数")
    print("="*60)
    print(f"  总参数量: {total_params:,}")
    print("="*60)

    # 确定输出路径
    if output_path is None:
        output_path = str(Path(model_pkl_path).parent / "lstm_weights.pth")

    # 保存权重
    torch.save(state_dict, output_path)
    print(f"\n✓ 权重已保存至: {output_path}")

    return output_path


def create_gat_workflow_script(lstm_weights_path):
    """
    创建使用LSTM权重初始化的GAT训练脚本
    """
    script_content = f'''"""
使用LSTM预训练权重训练GAT模型
"""

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.contrib.model.pytorch_gats import GATs

# 初始化Qlib
provider_uri = "~/.qlib/qlib_data/cn_data"  # 修改为你的数据路径
qlib.init(provider_uri=provider_uri, region=REG_CN)

# 数据集配置
market = "csi300"
benchmark = "SH000300"

data_handler_config = {{
    "start_time": "2008-01-01",
    "end_time": "2020-08-01",
    "fit_start_time": "2008-01-01",
    "fit_end_time": "2014-12-31",
    "instruments": market,
}}

# 创建数据集
dataset_config = {{
    "class": "DatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {{
        "handler": {{
            "class": "Alpha360",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": data_handler_config,
        }},
        "segments": {{
            "train": ("2008-01-01", "2014-12-31"),
            "valid": ("2015-01-01", "2016-12-31"),
            "test": ("2017-01-01", "2020-08-01"),
        }},
    }},
}}

dataset = init_instance_by_config(dataset_config)

# 创建GAT模型（使用LSTM权重初始化）
print("="*60)
print("创建GAT模型，使用LSTM预训练权重初始化")
print("="*60)

model = GATs(
    d_feat=6,              # Alpha360特征维度
    hidden_size=64,        # 隐藏层大小（需要与LSTM一致）
    num_layers=2,          # 层数（需要与LSTM一致）
    dropout=0.0,
    n_epochs=200,
    lr=0.001,
    early_stop=20,
    metric="loss",
    loss="mse",
    base_model="LSTM",     # 使用LSTM作为base模型
    model_path="{lstm_weights_path}",  # LSTM预训练权重路径
    optimizer="adam",
    GPU=0,
)

# 训练模型
print("\\n开始训练GAT模型...")
with R.start(experiment_name="gat_with_lstm_init"):
    R.log_params(
        base_model="LSTM",
        pretrained_weights="{lstm_weights_path}",
        d_feat=6,
        hidden_size=64,
        num_layers=2,
    )

    model.fit(dataset, save_path="gat_model.pth")

    # 预测
    pred = model.predict(dataset)

    # 记录结果
    R.save_objects(**{{"pred.pkl": pred}})

    # 信号分析
    sig_rec = SignalRecord(model=model, dataset=dataset, recorder=R)
    sig_rec.generate()

    # 回测配置
    port_analysis_config = {{
        "strategy": {{
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy",
            "kwargs": {{
                "signal": pred,
                "topk": 50,
                "n_drop": 5,
            }},
        }},
        "backtest": {{
            "start_time": "2017-01-01",
            "end_time": "2020-08-01",
            "account": 100000000,
            "benchmark": benchmark,
            "exchange_kwargs": {{
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            }},
        }},
    }}

    # 组合分析
    port_rec = PortAnaRecord(recorder=R, config=port_analysis_config)
    port_rec.generate()

print("\\n✓ 训练完成！")
'''

    output_file = "train_gat_with_lstm.py"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(script_content)

    print(f"\n✓ GAT训练脚本已创建: {output_file}")
    return output_file


def main():
    # LSTM模型路径
    lstm_model_path = "./mlruns/686249546704553370/2ee5b9c2d8f2420e8cb39aa84b35546e/artifacts/params.pkl"

    print("="*60)
    print("从Qlib LSTM模型提取权重用于GAT初始化")
    print("="*60)

    # 提取权重
    weights_path = extract_weights_from_qlib_lstm(lstm_model_path)

    if weights_path:
        # 创建训练脚本
        script_path = create_gat_workflow_script(weights_path)

        print("\n" + "="*60)
        print("使用方法:")
        print("="*60)
        print(f"\n1. 权重文件已保存至: {weights_path}")
        print(f"2. 训练脚本已创建: {script_path}")
        print(f"\n直接运行:")
        print(f"   python {script_path}")

        print("\n或者在你自己的代码中使用:")
        print(f"""
from qlib.contrib.model.pytorch_gats import GATs

model = GATs(
    d_feat=6,
    hidden_size=64,
    num_layers=2,
    base_model="LSTM",
    model_path="{weights_path}",  # LSTM权重路径
    lr=0.001,
    n_epochs=200,
)

model.fit(dataset, save_path="gat_model.pth")
""")


if __name__ == "__main__":
    main()
