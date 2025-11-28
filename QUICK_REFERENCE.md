# 快速参考卡片 - Qlib模型使用

## 🎯 你的需求 → 使用的工具

| 我想... | 使用这个 | 命令 |
|---------|----------|------|
| **提取LSTM权重给GAT用** | `extract_lstm_weights.py` | `python extract_lstm_weights.py` |
| **加载模型复现回测** | `load_and_predict.py` | `python load_and_predict.py` |
| **快速预测指定时间段** | `quick_predict.py` | `python quick_predict.py --start_date 2020-01-01 --end_date 2020-12-31` |
| **查看详细使用文档** | `MODEL_USAGE_GUIDE.md` | 直接阅读 |
| **了解所有脚本功能** | `README_SCRIPTS.md` | 直接阅读 |

---

## 💡 最常用的3个命令

### 1️⃣ 提取LSTM权重 → 训练GAT
```bash
# 一键提取并生成训练脚本
python extract_lstm_weights.py

# 输出:
# ✓ lstm_weights.pth (权重文件)
# ✓ train_gat_with_lstm.py (训练脚本)

# 然后训练GAT
python train_gat_with_lstm.py
```

### 2️⃣ 加载模型 → 复现回测
```bash
python load_and_predict.py

# 输出:
# ✓ predictions_original_test.csv (原始测试集预测)
# ✓ predictions_new_data.csv (新数据预测)
# ✓ MLflow实验记录 (完整回测报告)
```

### 3️⃣ 快速预测新数据
```bash
python quick_predict.py \
    --model_path ./mlruns/686249546704553370/2ee5b9c2d8f2420e8cb39aa84b35546e/artifacts/params.pkl \
    --start_date 2021-01-01 \
    --end_date 2021-12-31 \
    --output predictions_2021.csv
```

---

## 📦 Python代码片段

### 加载模型并预测
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
                "end_time": "2021-12-31",
                "instruments": "csi300",
            },
        },
        "segments": {
            "test": ("2021-01-01", "2021-12-31"),
        },
    },
}
dataset = init_instance_by_config(dataset_config)

# 预测
pred = model.predict(dataset, segment="test")
pred.to_csv("predictions.csv")
```

### GAT使用LSTM权重
```python
from qlib.contrib.model.pytorch_gats import GATs

model = GATs(
    d_feat=6,
    hidden_size=64,
    num_layers=2,
    base_model="LSTM",
    model_path="lstm_weights.pth",  # ← LSTM权重路径
    lr=0.001,
    n_epochs=200,
)

model.fit(dataset, save_path="gat_model.pth")
```

---

## 🔍 常见问题快速解答

**Q: 模型文件在哪？**
```bash
find ./mlruns -name "params.pkl" -mtime -1
```

**Q: 如何查看回测结果？**
```bash
qlib_mlflow  # 打开MLflow UI
```

**Q: 预测结果是什么格式？**
```
datetime,instrument,score
2020-01-02,SH600000,0.123456
2020-01-02,SH600001,-0.234567
```

**Q: 权重文件包含什么？**
```
rnn.weight_ih_l0, rnn.weight_hh_l0  # LSTM层权重
rnn.bias_ih_l0, rnn.bias_hh_l0      # LSTM层偏置
fc_out.weight, fc_out.bias          # 输出层权重
```

---

## 🚦 工作流程图

```
训练LSTM模型
    ↓
保存到 mlruns/.../params.pkl
    ↓
    ├─→ [选项1] 直接使用
    │       ↓
    │   load_and_predict.py → 复现回测 + 新数据预测
    │
    ├─→ [选项2] 提取权重
    │       ↓
    │   extract_lstm_weights.py → lstm_weights.pth
    │       ↓
    │   train_gat_with_lstm.py → 训练GAT
    │
    └─→ [选项3] 快速预测
            ↓
        quick_predict.py → predictions.csv
```

---

## 📞 需要帮助？

查看详细文档：
- `MODEL_USAGE_GUIDE.md` - 完整使用指南
- `README_SCRIPTS.md` - 所有脚本说明

Qlib官方文档：
- https://qlib.readthedocs.io/

---

**版本**: v1.0
**更新时间**: 2025-11-28
**适用于**: Qlib + PyTorch模型
