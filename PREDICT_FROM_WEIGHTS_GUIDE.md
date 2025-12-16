# 从权重文件加载模型进行预测 - 使用指南

## 问题背景

训练完成后，模型权重保存在 `gats_model.pth`，但 MLflow artifacts 目录为空，导致无法使用原来的 `load_and_predict.py` 脚本（它依赖 MLflow 的 `params.pkl`）。

## 解决方案

使用新脚本 `predict_from_weights.py`，直接从 `.pth` 权重文件加载模型。

## 文件说明

### 1. `predict_from_weights.py` - 主预测脚本

**功能**：
- 从 `gats_model.pth` 权重文件加载模型
- 对指定时间段数据进行推理
- 执行回测分析
- 保存预测结果为 CSV 文件

**使用方法**：

```bash
# 在 Ubuntu 服务器上运行
cd /data10/tyc/quant/qlib_own
python predict_from_weights.py
```

**配置选项**（在脚本的 `main()` 函数中修改）：

```python
# 1. 路径配置（自动检测系统）
WEIGHTS_PATH = "gats_model.pth"  # 权重文件
CONFIG_PATH = "examples/benchmarks/GATs/workflow_config_gats_Alpha158.yaml"  # 配置文件

# 2. 预测时间段配置
ORIGINAL_TEST_SEGMENT = ("2024-01-01", "2024-12-31")  # 原始测试集
ORIGINAL_OUTPUT = "predictions_2024.csv"  # 输出文件

NEW_TEST_SEGMENT = ("2025-08-09", "2025-11-21")  # 新数据
NEW_OUTPUT = "predictions_2025.csv"  # 输出文件

# 3. 是否执行回测
RUN_BACKTEST = True  # 设为 False 可以跳过回测，只生成预测
```

### 2. `add_model_to_mlflow.py` - 补充 MLflow artifacts

**功能**：
- 将已训练的模型权重添加到 MLflow 实验记录中

**使用方法**：

```bash
python add_model_to_mlflow.py
```

**注意**：现在已修复路径问题，可以在 Ubuntu 上正常运行。

### 3. `train_gats_enhanced_example.py` - 训练脚本（已更新）

**更新内容**：
- 训练完成后自动将模型权重保存到 MLflow artifacts
- 下次训练时会自动保存 `gats_model.pth` 和 `best_checkpoint.pth` 到 MLflow

## 输出文件格式

生成的 CSV 文件格式（与 `load_and_predict.py` 兼容）：

```csv
datetime,instrument,score
2024-01-02,SH000300,0.074863
2024-01-03,SH000300,0.094997
2024-01-04,SH000300,0.147436
...
```

列说明：
- `datetime`: 日期
- `instrument`: 股票代码
- `score`: 预测分数（越高表示预期收益越好）

## 典型工作流

### 场景 1: 对已训练模型进行预测

```bash
# 1. 直接运行预测脚本
python predict_from_weights.py

# 2. 查看生成的预测文件
head predictions_2024.csv
head predictions_2025.csv
```

### 场景 2: 修改预测时间段

编辑 `predict_from_weights.py`，修改：

```python
# 例如：预测 2025 年全年
ORIGINAL_TEST_SEGMENT = ("2025-01-01", "2025-12-31")
ORIGINAL_OUTPUT = "predictions_2025_full_year.csv"
```

然后运行：

```bash
python predict_from_weights.py
```

### 场景 3: 只生成预测，跳过回测

编辑 `predict_from_weights.py`，设置：

```python
RUN_BACKTEST = False  # 跳过回测（加快速度）
```

### 场景 4: 补充 MLflow artifacts（可选）

如果希望在 MLflow UI 中也能看到模型权重：

```bash
python add_model_to_mlflow.py
```

然后查看：

```bash
ls -lh mlruns/789877461266984949/f8a1507c4a0d4499b7e70061966557fd/artifacts/
```

## 回测结果示例

运行脚本后会输出类似以下的回测结果：

```
============================================================
Backtest Results (Frequency: day)
============================================================

【Benchmark Performance】
  Annualized Return: 0.1704 (17.04%)
  Information Ratio:  0.9135
  Max Drawdown:       -0.1579 (-15.79%)

【Excess Return (Without Cost)】
  Annualized Return: 0.7935 (79.35%)
  Information Ratio:  3.5414
  Max Drawdown:       -0.2092 (-20.92%)

【Excess Return (With Cost)】
  Annualized Return: 0.7522 (75.22%)
  Information Ratio:  3.3584
  Max Drawdown:       -0.2118 (-21.18%)

【Key Metrics Summary】
  Strategy Annualized Return: 0.7522 (75.22%)
  Strategy Sharpe Ratio:      3.3584
  Strategy Max Drawdown:      -0.2118 (-21.18%)
```

## 故障排除

### 问题 1: FileNotFoundError

**错误**：`FileNotFoundError: gats_model.pth`

**解决**：检查权重文件路径是否正确

```bash
# 检查文件是否存在
ls -lh gats_model.pth

# 如果文件在其他位置，修改脚本中的 WEIGHTS_PATH
```

### 问题 2: 数据不存在

**错误**：数据加载失败或预测结果为空

**解决**：检查 Qlib 数据是否包含指定时间段

```bash
# 检查数据范围
python check_data_range.py
```

### 问题 3: CUDA/GPU 错误

**错误**：CUDA out of memory 或 GPU 相关错误

**解决**：脚本已默认使用 CPU，无需修改。如果仍有问题，确认：

```python
# 在脚本中确认这行：
GPU=-1,  # 使用 CPU
```

## 下次训练时的建议

下次运行训练脚本时，已更新的 `train_gats_enhanced_example.py` 会自动：
1. 保存 `gats_model.pth` 到本地
2. 保存 `gats_model.pth` 到 MLflow artifacts
3. 保存 `best_checkpoint.pth` 到 MLflow artifacts/checkpoints/

这样就不需要再手动添加到 MLflow 了。

## 总结

- **主要脚本**：`predict_from_weights.py` - 从权重文件加载模型并预测
- **辅助脚本**：`add_model_to_mlflow.py` - 补充 MLflow artifacts（可选）
- **输出格式**：CSV 文件，与原 `load_and_predict.py` 兼容
- **系统兼容**：自动检测 macOS/Linux，无需修改路径
