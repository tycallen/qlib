# 模型加载、推理与回测使用指南

本指南介绍如何使用训练好的模型进行推理、复现回测结果以及在新数据上进行预测。

## 目录

1. [模型文件位置](#模型文件位置)
2. [快速使用](#快速使用)
3. [完整使用流程](#完整使用流程)
4. [常见场景](#常见场景)
5. [GAT模型使用LSTM权重](#gat模型使用lstm权重)

---

## 模型文件位置

训练完成后，模型文件保存在MLflow实验目录中：

```
./mlruns/<实验ID>/<运行ID>/artifacts/params.pkl
```

例如：
```
./mlruns/686249546704553370/2ee5b9c2d8f2420e8cb39aa84b35546e/artifacts/params.pkl
```

查找最新的模型：
```bash
# 查找最近修改的模型文件
find ./mlruns -name "params.pkl" -type f -mtime -1

# 查看实验记录
qlib_mlflow
```

---

## 快速使用

### 方式1: 使用快速预测脚本

```bash
python quick_predict.py \
    --model_path ./mlruns/686249546704553370/2ee5b9c2d8f2420e8cb39aa84b35546e/artifacts/params.pkl \
    --start_date 2020-01-01 \
    --end_date 2020-12-31 \
    --output predictions.csv
```

### 方式2: 使用Python代码

```python
import pickle
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config

# 初始化Qlib
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

# 加载模型
with open('params.pkl', 'rb') as f:
    model = pickle.load(f)

# 创建数据集
dataset_config = {
    "class": "DatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {
        "handler": {
            "class": "Alpha360",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "start_time": "2008-01-01",
                "end_time": "2020-12-31",
                "fit_start_time": "2008-01-01",
                "fit_end_time": "2014-12-31",
                "instruments": "csi300",
            },
        },
        "segments": {
            "test": ("2020-01-01", "2020-12-31"),
        },
    },
}
dataset = init_instance_by_config(dataset_config)

# 预测
pred = model.predict(dataset, segment="test")

# 保存结果
pred.to_csv("predictions.csv")
```

---

## 完整使用流程

### 1. 复现训练时的回测结果

```bash
python load_and_predict.py
```

这个脚本会：
- 加载训练好的模型
- 在原始测试集上进行预测
- 复现回测结果
- 生成信号分析和组合分析报告

### 2. 对新数据进行推理

修改 `load_and_predict.py` 中的配置：

```python
# 修改这部分代码
NEW_TEST_SEGMENT = ("2021-01-01", "2022-12-31")  # 改为你想要的时间段
```

然后运行：
```bash
python load_and_predict.py
```

---

## 常见场景

### 场景1: 加载模型权重到新模型

```python
import torch
from qlib.contrib.model.pytorch_lstm import LSTM, LSTMModel

# 方式1: 加载完整的Qlib模型对象
import pickle
with open('params.pkl', 'rb') as f:
    trained_model = pickle.load(f)

# 方式2: 只加载PyTorch权重
# 先从Qlib模型对象中提取权重
state_dict = trained_model.LSTM_model.state_dict()
torch.save(state_dict, 'lstm_weights.pth')

# 然后加载到新模型
new_model = LSTMModel(d_feat=6, hidden_size=64, num_layers=2)
new_model.load_state_dict(torch.load('lstm_weights.pth'))
```

### 场景2: 批量预测多个时间段

```python
import pickle
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config

# 初始化
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

# 加载模型
with open('params.pkl', 'rb') as f:
    model = pickle.load(f)

# 定义多个时间段
time_periods = [
    ("2020-01-01", "2020-06-30"),
    ("2020-07-01", "2020-12-31"),
    ("2021-01-01", "2021-06-30"),
    ("2021-07-01", "2021-12-31"),
]

# 批量预测
all_predictions = []
for start, end in time_periods:
    print(f"预测时间段: {start} - {end}")

    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha360",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": "2008-01-01",
                    "end_time": end,
                    "fit_start_time": "2008-01-01",
                    "fit_end_time": "2014-12-31",
                    "instruments": "csi300",
                },
            },
            "segments": {
                "test": (start, end),
            },
        },
    }

    dataset = init_instance_by_config(dataset_config)
    pred = model.predict(dataset, segment="test")
    all_predictions.append(pred)

    # 保存每个时间段的预测
    pred.to_csv(f"predictions_{start}_{end}.csv")

# 合并所有预测
import pandas as pd
combined_pred = pd.concat(all_predictions)
combined_pred.to_csv("predictions_combined.csv")
```

### 场景3: 提取模型权重用于其他用途

```python
# 运行提取脚本
python extract_lstm_weights.py
```

这会生成：
- `lstm_weights.pth`: PyTorch格式的权重文件
- 权重详细信息报告

---

## GAT模型使用LSTM权重

### 步骤1: 提取LSTM权重

```bash
python extract_lstm_weights.py
```

### 步骤2: 使用LSTM权重训练GAT

```python
from qlib.contrib.model.pytorch_gats import GATs

model = GATs(
    d_feat=6,
    hidden_size=64,
    num_layers=2,
    dropout=0.0,
    base_model="LSTM",  # 指定使用LSTM作为base模型
    model_path="./mlruns/.../artifacts/lstm_weights.pth",  # LSTM权重路径
    lr=0.001,
    n_epochs=200,
)

# 训练（会自动加载LSTM权重初始化）
model.fit(dataset, save_path="gat_model.pth")
```

或使用生成的训练脚本：
```bash
python train_gat_with_lstm.py
```

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `load_and_predict.py` | 完整的加载、复现回测、新数据推理脚本 |
| `quick_predict.py` | 快速预测脚本 |
| `extract_lstm_weights.py` | 提取LSTM权重用于GAT |
| `train_gat_with_lstm.py` | 使用LSTM权重训练GAT（自动生成） |

---

## 常见问题

### Q1: 如何找到训练好的模型？

```bash
# 查看所有实验
qlib_mlflow

# 或查找最近的模型文件
find ./mlruns -name "params.pkl" -mtime -1
```

### Q2: 预测结果包含NaN怎么办？

检查：
1. 数据时间范围是否正确
2. 股票代码（instruments）是否有数据
3. 特征计算是否需要历史数据

### Q3: 如何查看回测报告？

```python
# 在脚本中运行回测后，查看实验记录
import qlib
from qlib.workflow import R

# 加载实验记录
rec = R.get_recorder(recorder_id="<实验ID>")

# 查看报告
report = rec.load_object("portfolio_analysis/report_normal_1day.pkl")
print(report)
```

### Q4: 如何修改回测参数？

在 `load_and_predict.py` 中修改 `backtest_config`：

```python
backtest_config = {
    "strategy": {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy",
        "kwargs": {
            "signal": None,
            "topk": 50,        # 修改选股数量
            "n_drop": 5,       # 修改每天剔除数量
        },
    },
    "backtest": {
        "account": 100000000,  # 修改初始资金
        "open_cost": 0.0005,   # 修改交易成本
        "close_cost": 0.0015,
        # ...
    },
}
```

---

## 进阶使用

### 自定义预测处理

```python
import pickle
import qlib
from qlib.constant import REG_CN

# 加载模型
with open('params.pkl', 'rb') as f:
    model = pickle.load(f)

# 获取原始预测
pred = model.predict(dataset, segment="test")

# 后处理：标准化
pred_normalized = (pred - pred.mean()) / pred.std()

# 后处理：截断极值
pred_clipped = pred.clip(lower=pred.quantile(0.05),
                         upper=pred.quantile(0.95))

# 后处理：排名
pred_rank = pred.groupby(level=0).rank(pct=True)

# 使用处理后的预测进行回测
backtest_config["strategy"]["kwargs"]["signal"] = pred_rank
```

### 集成多个模型

```python
import pickle

# 加载多个模型
model1 = pickle.load(open('lstm_model.pkl', 'rb'))
model2 = pickle.load(open('gru_model.pkl', 'rb'))

# 分别预测
pred1 = model1.predict(dataset, segment="test")
pred2 = model2.predict(dataset, segment="test")

# 集成（简单平均）
pred_ensemble = (pred1 + pred2) / 2

# 集成（加权平均）
pred_ensemble = 0.6 * pred1 + 0.4 * pred2

# 使用集成结果回测
backtest_config["strategy"]["kwargs"]["signal"] = pred_ensemble
```

---

## 总结

关键步骤：
1. **加载模型**: 使用 `pickle.load()` 或提供的脚本
2. **准备数据**: 使用 `DatasetH` 和对应的 `Handler`
3. **预测**: `model.predict(dataset, segment="test")`
4. **回测**: 使用 `PortAnaRecord` 生成回测报告
5. **保存结果**: 保存为CSV或使用MLflow记录

需要帮助请查看Qlib文档: https://qlib.readthedocs.io/
