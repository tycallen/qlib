"""
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

data_handler_config = {
    "start_time": "2008-01-01",
    "end_time": "2020-08-01",
    "fit_start_time": "2008-01-01",
    "fit_end_time": "2014-12-31",
    "instruments": market,
}

# 创建数据集
dataset_config = {
    "class": "DatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {
        "handler": {
            "class": "Alpha360",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": data_handler_config,
        },
        "segments": {
            "train": ("2008-01-01", "2014-12-31"),
            "valid": ("2015-01-01", "2016-12-31"),
            "test": ("2017-01-01", "2020-08-01"),
        },
    },
}

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
    model_path="mlruns/686249546704553370/2ee5b9c2d8f2420e8cb39aa84b35546e/artifacts/lstm_weights.pth",  # LSTM预训练权重路径
    optimizer="adam",
    GPU=0,
)

# 训练模型
print("\n开始训练GAT模型...")
with R.start(experiment_name="gat_with_lstm_init"):
    R.log_params(
        base_model="LSTM",
        pretrained_weights="mlruns/686249546704553370/2ee5b9c2d8f2420e8cb39aa84b35546e/artifacts/lstm_weights.pth",
        d_feat=6,
        hidden_size=64,
        num_layers=2,
    )

    model.fit(dataset, save_path="gat_model.pth")

    # 预测
    pred = model.predict(dataset)

    # 记录结果
    R.save_objects(**{"pred.pkl": pred})

    # 信号分析
    sig_rec = SignalRecord(model=model, dataset=dataset, recorder=R)
    sig_rec.generate()

    # 回测配置
    port_analysis_config = {
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

    # 组合分析
    port_rec = PortAnaRecord(recorder=R, config=port_analysis_config)
    port_rec.generate()

print("\n✓ 训练完成！")
