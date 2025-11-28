# Qlib模型使用脚本集合

本目录包含了一系列用于Qlib模型训练、推理、权重提取和回测的实用脚本。

## 📁 文件清单

| 文件 | 用途 | 使用场景 |
|------|------|----------|
| `extract_lstm_weights.py` | 从Qlib LSTM模型提取PyTorch权重 | 为GAT模型准备预训练权重 |
| `load_and_predict.py` | 完整的模型加载、回测复现和新数据推理 | 评估模型性能、在新数据上预测 |
| `quick_predict.py` | 快速预测脚本（命令行工具） | 快速对指定时间段进行预测 |
| `train_gat_with_lstm.py` | 使用LSTM权重训练GAT（自动生成） | 迁移学习 |
| `MODEL_USAGE_GUIDE.md` | 详细使用指南 | 参考文档 |

## 🚀 快速开始

### 1. 提取LSTM权重用于GAT

你已经训练好了LSTM模型，现在想将其权重用于GAT模型的初始化。

```bash
# 提取LSTM权重
python extract_lstm_weights.py

# 输出：
# - lstm_weights.pth: PyTorch权重文件
# - train_gat_with_lstm.py: GAT训练脚本

# 使用LSTM权重训练GAT
python train_gat_with_lstm.py
```

**工作原理**：
1. 从Qlib的LSTM模型对象（`params.pkl`）中提取PyTorch模型的`state_dict`
2. 保存为标准的`.pth`格式
3. GAT模型在初始化时加载这些权重（`rnn.*`层会被复用）

### 2. 加载模型并复现回测结果

你想验证训练好的模型在测试集上的表现。

```bash
python load_and_predict.py
```

**功能**：
- ✅ 加载训练好的模型
- ✅ 在原始测试集上预测
- ✅ 复现训练时的回测结果
- ✅ 生成信号分析报告
- ✅ 生成组合回测报告

### 3. 对新数据快速推理

你想对新的时间段进行预测（例如2021年的数据）。

```bash
python quick_predict.py \
    --model_path ./mlruns/686249546704553370/2ee5b9c2d8f2420e8cb39aa84b35546e/artifacts/params.pkl \
    --start_date 2021-01-01 \
    --end_date 2021-12-31 \
    --output predictions_2021.csv
```

**输出**：
- `predictions_2021.csv`: CSV格式的预测结果

## 📖 详细使用说明

### extract_lstm_weights.py

**功能**：从Qlib的LSTM模型中提取PyTorch权重

**输入**：
- Qlib保存的LSTM模型文件（`params.pkl`）

**输出**：
- `lstm_weights.pth`: PyTorch格式的权重文件
- `train_gat_with_lstm.py`: 使用LSTM权重的GAT训练脚本

**使用方法**：
```python
# 脚本内已配置好路径，直接运行
python extract_lstm_weights.py

# 或在Python中调用
from extract_lstm_weights import extract_weights_from_qlib_lstm

weights_path = extract_weights_from_qlib_lstm(
    model_pkl_path="./mlruns/.../params.pkl",
    output_path="my_lstm_weights.pth"
)
```

**权重结构**：
```
LSTM模型权重包含:
  - rnn.weight_ih_l0    # LSTM输入权重（第0层）
  - rnn.weight_hh_l0    # LSTM隐藏状态权重（第0层）
  - rnn.bias_ih_l0      # LSTM输入偏置（第0层）
  - rnn.bias_hh_l0      # LSTM隐藏状态偏置（第0层）
  - rnn.weight_ih_l1    # LSTM输入权重（第1层）
  - rnn.weight_hh_l1    # LSTM隐藏状态权重（第1层）
  - rnn.bias_ih_l1      # LSTM输入偏置（第1层）
  - rnn.bias_hh_l1      # LSTM隐藏状态偏置（第1层）
  - fc_out.weight       # 输出层权重
  - fc_out.bias         # 输出层偏置
```

---

### load_and_predict.py

**功能**：完整的模型加载、回测复现和新数据推理流程

**主要功能**：
1. 加载训练好的模型
2. 复现原始测试集的回测结果
3. 对新时间段数据进行推理
4. 生成完整的分析报告

**配置项**（在脚本中修改）：
```python
# 模型路径
MODEL_PKL_PATH = "./mlruns/686249546704553370/.../params.pkl"

# 数据集配置
dataset_config = {
    "kwargs": {
        "handler": {
            "kwargs": {
                "instruments": "csi300",  # 股票池
                # ...
            }
        },
        "segments": {
            "test": ("2017-01-01", "2020-08-01"),  # 测试时间段
        },
    },
}

# 回测配置
backtest_config = {
    "strategy": {
        "kwargs": {
            "topk": 50,      # 选股数量
            "n_drop": 5,     # 每天剔除数量
        },
    },
    "backtest": {
        "account": 100000000,  # 初始资金
        # ...
    },
}

# 新数据时间段
NEW_TEST_SEGMENT = ("2020-08-01", "2021-12-31")
```

**输出**：
- `predictions_original_test.csv`: 原始测试集预测
- `predictions_new_data.csv`: 新数据预测
- MLflow实验记录（包含回测报告）

---

### quick_predict.py

**功能**：快速命令行预测工具

**命令行参数**：
```bash
--model_path    # 模型pkl文件路径
--start_date    # 预测开始日期（格式：YYYY-MM-DD）
--end_date      # 预测结束日期（格式：YYYY-MM-DD）
--output        # 输出CSV文件路径
```

**示例**：
```bash
# 基本使用
python quick_predict.py \
    --model_path ./mlruns/686249546704553370/2ee5b9c2d8f2420e8cb39aa84b35546e/artifacts/params.pkl \
    --start_date 2020-01-01 \
    --end_date 2020-12-31

# 指定输出文件
python quick_predict.py \
    --model_path params.pkl \
    --start_date 2021-01-01 \
    --end_date 2021-12-31 \
    --output pred_2021.csv
```

---

## 🔧 使用场景示例

### 场景1: 训练GAT模型使用LSTM权重初始化

```bash
# 步骤1: 提取LSTM权重
python extract_lstm_weights.py

# 步骤2: 训练GAT（会自动生成脚本）
python train_gat_with_lstm.py
```

### 场景2: 验证模型在测试集上的表现

```bash
# 运行完整的回测流程
python load_and_predict.py

# 查看MLflow UI
qlib_mlflow
```

### 场景3: 批量预测多个时间段

```bash
# 2020年上半年
python quick_predict.py \
    --model_path params.pkl \
    --start_date 2020-01-01 \
    --end_date 2020-06-30 \
    --output pred_2020_h1.csv

# 2020年下半年
python quick_predict.py \
    --model_path params.pkl \
    --start_date 2020-07-01 \
    --end_date 2020-12-31 \
    --output pred_2020_h2.csv

# 2021年全年
python quick_predict.py \
    --model_path params.pkl \
    --start_date 2021-01-01 \
    --end_date 2021-12-31 \
    --output pred_2021.csv
```

### 场景4: 在Python代码中使用

```python
# 导入工具函数
from load_and_predict import load_trained_model, predict_new_data
import qlib
from qlib.constant import REG_CN

# 初始化
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

# 加载模型
model = load_trained_model("params.pkl")

# 预测
dataset_config = {...}  # 你的配置
pred = predict_new_data(model, dataset_config, ("2021-01-01", "2021-12-31"))

# 保存
pred.to_csv("my_predictions.csv")
```

---

## 📊 预测结果格式

所有脚本生成的CSV文件格式：

```csv
datetime,instrument,score
2020-01-02,SH600000,0.123456
2020-01-02,SH600001,-0.234567
2020-01-02,SH600002,0.345678
...
```

**字段说明**：
- `datetime`: 日期
- `instrument`: 股票代码
- `score`: 预测分数（越高越好）

---

## 🐛 故障排除

### 问题1: 找不到模型文件

```bash
# 查找最近的模型
find ./mlruns -name "params.pkl" -mtime -1

# 列出所有实验
ls -la ./mlruns/*/
```

### 问题2: LSTM模型属性名不匹配

错误信息：`无法找到lstm_model或LSTM_model属性`

**解决**：
- `pytorch_lstm.py` 使用 `lstm_model`（小写）
- `pytorch_lstm_ts.py` 使用 `LSTM_model`（大写）
- 脚本已自动处理两种情况

### 问题3: 数据不存在

错误信息：`Empty data from dataset`

**检查**：
1. Qlib数据是否已下载：`ls ~/.qlib/qlib_data/cn_data/`
2. 时间范围是否超出数据范围
3. 股票池配置是否正确

### 问题4: GPU内存不足

如果使用GPU训练时内存不足，可以：
```python
# 在模型配置中设置
model = GATs(
    ...
    GPU=-1,  # 使用CPU
    batch_size=400,  # 减小batch size
)
```

---

## 📚 相关文档

- [MODEL_USAGE_GUIDE.md](MODEL_USAGE_GUIDE.md) - 详细使用指南
- [Qlib官方文档](https://qlib.readthedocs.io/)
- [PyTorch权重加载文档](https://pytorch.org/tutorials/beginner/saving_loading_models.html)

---

## ✨ 最佳实践

1. **模型管理**
   - 训练时使用有意义的实验名称
   - 在MLflow中记录关键参数
   - 定期备份重要模型

2. **推理优化**
   - 对于大规模预测，考虑批处理
   - 使用GPU加速（如果可用）
   - 缓存数据集避免重复加载

3. **结果验证**
   - 检查预测分数的分布
   - 对比不同时间段的表现
   - 进行样本外测试

4. **权重复用**
   - LSTM和GRU权重可以互相迁移（调整后）
   - GAT可以使用LSTM或GRU的RNN层权重
   - 保存中间检查点便于调试

---

## 🤝 贡献

如果你有改进建议或发现bug，欢迎提Issue或PR。

## 📄 许可

MIT License - 基于Qlib项目
