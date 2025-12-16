# 🚀 推理快速开始指南

## 问题
训练时用了3张显卡，推理时OOM（显存不足）+ 权重加载失败

## 最新更新 (2025-12-16)
✅ 已修复多GPU权重加载问题
- 修复了 DataParallel 包装顺序导致的权重加载失败
- 现在可以正确加载训练时保存的权重并使用多GPU推理

## 解决方案

### ⭐ 方法1: 使用配置脚本（最简单）

```bash
cd /data10/tyc/quant/qlib_own

# 配置多GPU推理（推荐）
python configure_inference.py --mode multi_gpu --gpus 0,1,2

# 或配置CPU推理（安全但慢）
python configure_inference.py --mode cpu

# 运行推理
python predict_from_weights.py
```

---

### 方法2: 手动修改配置

编辑 `predict_from_weights.py`，找到第315行左右，修改：

#### 选项A: 多GPU（推荐，快速）

```python
# 3. 推理模式配置
INFERENCE_MODE = "multi_gpu"
INFERENCE_GPU = 0
INFERENCE_GPUS = [0, 1, 2]  # 使用3张显卡
```

#### 选项B: CPU（安全，较慢）

```python
# 3. 推理模式配置
INFERENCE_MODE = "cpu"
INFERENCE_GPU = -1
INFERENCE_GPUS = None
```

然后运行：
```bash
python predict_from_weights.py
```

---

## 配置对比

| 模式 | 速度 | 显存需求 | 命令 |
|------|------|---------|------|
| **多GPU (推荐)** | ⚡ 最快 (1-3分钟) | 3张GPU | `--mode multi_gpu --gpus 0,1,2` |
| **CPU** | 🐌 较慢 (10-30分钟) | 不需要GPU | `--mode cpu` |
| **单GPU** | ⚠️ 会OOM | 需要>12GB | 不推荐 |

---

## 完整流程示例

### 场景1: 使用多GPU（推荐）

```bash
# 1. 检查GPU状态
nvidia-smi

# 2. 配置多GPU推理
python configure_inference.py --mode multi_gpu --gpus 0,1,2

# 3. 运行推理
python predict_from_weights.py

# 4. 查看结果
head predictions_2024.csv
head predictions_2025.csv
```

**预期输出**：
```
GPU Configuration:
  Mode: Multi-GPU
  GPUs: [0, 1, 2]
✓ Model wrapped with DataParallel on GPUs: [0, 1, 2]

Predicting on test set...
✓ Prediction completed
  Total samples: 61244

Predictions saved to: predictions_2024.csv
```

---

### 场景2: 使用CPU（GPU被占用时）

```bash
# 1. 配置CPU推理
python configure_inference.py --mode cpu

# 2. 运行推理（需要等待较长时间）
python predict_from_weights.py

# 3. 查看结果
head predictions_2024.csv
```

---

## 验证配置

运行后看到以下输出说明配置正确：

### 多GPU模式
```
GPU Configuration:
  Mode: Multi-GPU
  GPUs: [0, 1, 2]            ✓ 确认使用3张GPU
✓ Model wrapped with DataParallel on GPUs: [0, 1, 2]
```

### CPU模式
```
GPU Configuration:
  Mode: CPU                  ✓ 确认使用CPU
✓ Model moved to: cpu
```

---

## 监控运行

### 监控GPU使用（多GPU模式）
```bash
# 实时监控
watch -n 1 nvidia-smi

# 预期看到3张GPU都在使用
```

### 监控CPU使用（CPU模式）
```bash
# 查看CPU占用
htop

# 或
top
```

---

## 输出文件

成功运行后会生成：

```
predictions_2024.csv  - 2024年预测结果
predictions_2025.csv  - 2025年预测结果
```

文件格式：
```csv
datetime,instrument,score
2024-01-02,SH000300,0.074863
2024-01-03,SH000300,0.094997
...
```

---

## 常见问题

### Q: 配置了多GPU还是OOM？
**A**: 检查GPU是否被其他进程占用
```bash
nvidia-smi  # 查看GPU占用情况
```

### Q: CPU模式太慢？
**A**: 这是正常的，CPU推理速度确实慢很多
- 多GPU: 1-3分钟
- CPU: 10-30分钟

可以减少数据量测试：
```python
# 在 predict_from_weights.py 中修改
ORIGINAL_TEST_SEGMENT = ("2024-01-01", "2024-01-31")  # 只测试1个月
```

### Q: 想使用GPU 1和2（GPU 0被占用）？
**A**:
```bash
python configure_inference.py --mode multi_gpu --gpus 1,2
```

---

## 帮助文档

详细文档：
- **OOM问题详解**: `INFERENCE_OOM_SOLUTION.md`
- **推理脚本指南**: `PREDICT_FROM_WEIGHTS_GUIDE.md`

查看配置脚本帮助：
```bash
python configure_inference.py --help
```

---

## 总结

**推荐配置**（针对你的3卡训练环境）：

```bash
# 一键配置+运行
python configure_inference.py --mode multi_gpu --gpus 0,1,2
python predict_from_weights.py
```

**预计时间**: 1-3分钟
**生成文件**: predictions_2024.csv, predictions_2025.csv

如果GPU被占用，临时使用CPU：
```bash
python configure_inference.py --mode cpu
python predict_from_weights.py
```

**预计时间**: 10-30分钟
