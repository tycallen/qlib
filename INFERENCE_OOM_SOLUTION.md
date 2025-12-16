# 推理时显存 OOM 问题解决方案

## 问题分析

**错误信息**：
```
OutOfMemoryError: CUDA out of memory. Tried to allocate 11.90 GiB.
GPU 0 has a total capacity of 9.77 GiB
```

**原因**：
- 训练时使用了 **3 张显卡**（通过 DataParallel 分布式训练）
- 推理时默认只使用 **单张 GPU 0**
- 单张显卡的 9.77 GiB 显存无法容纳整个模型的推理计算（需要 11.90 GiB）

## 解决方案（3种）

### ⭐ 方案1: 多GPU推理（推荐，与训练一致）

**优点**：
- 速度快
- 与训练时环境一致
- 显存充足

**配置方法**：

编辑 `predict_from_weights.py`，在 `main()` 函数中修改：

```python
# 3. 推理模式配置
# 选项3: 多GPU模式（推荐，与训练时一致）
INFERENCE_MODE = "multi_gpu"
INFERENCE_GPU = 0
INFERENCE_GPUS = [0, 1, 2]  # 使用3张显卡
```

然后运行：

```bash
python predict_from_weights.py
```

**预期输出**：
```
GPU Configuration:
  Mode: Multi-GPU
  GPUs: [0, 1, 2]
✓ Model wrapped with DataParallel on GPUs: [0, 1, 2]
```

---

### 方案2: CPU推理（最安全，但慢）

**优点**：
- 不占用 GPU 显存
- 永远不会 OOM
- 适合测试和调试

**缺点**：
- **速度较慢**（可能需要 10-30 分钟）

**配置方法**：

编辑 `predict_from_weights.py`：

```python
# 3. 推理模式配置
# 选项1: CPU模式（最安全，但较慢）
INFERENCE_MODE = "cpu"
INFERENCE_GPU = -1
INFERENCE_GPUS = None
```

**优化建议**：
```python
# 增加 CPU 线程数加速
model_config['n_jobs'] = 20  # 根据CPU核心数调整
```

---

### 方案3: 单GPU + 批处理推理（需要代码修改）

如果你只想使用单张 GPU，需要修改模型的 `predict` 方法，使用更小的 batch size。

**实现步骤**：

1. 修改 `pytorch_gats_ts_optimized.py` 的 `predict` 方法
2. 使用更小的 batch size 进行推理
3. 分批处理后合并结果

**示例代码**（需要添加到模型中）：

```python
def predict(self, dataset, batch_size=32):
    """
    推理时使用小批量以减少显存占用
    """
    # ... 原有代码 ...

    # 使用更小的batch size
    test_loader = DataLoader(
        dl_test,
        batch_size=batch_size,  # 减小batch size
        shuffle=False,
        num_workers=self.n_jobs,
    )

    # 分批推理
    predictions = []
    for batch in test_loader:
        with torch.no_grad():
            pred = self.GAT_model(batch).detach().cpu()
            predictions.append(pred)

    return torch.cat(predictions)
```

---

## 快速解决步骤

### Step 1: 检查GPU可用性

```bash
# 检查GPU状态
nvidia-smi

# 查看哪些GPU空闲
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
```

### Step 2: 选择推理模式

根据你的情况选择：

| 场景 | 推荐方案 | 配置 |
|------|---------|------|
| **有3张GPU空闲** | 方案1: 多GPU | `INFERENCE_GPUS = [0, 1, 2]` |
| **有1张GPU空闲但显存不够** | 方案2: CPU | `INFERENCE_GPU = -1` |
| **时间紧急** | 方案1: 多GPU | 速度最快 |
| **不急，想省GPU** | 方案2: CPU | 不占用GPU |

### Step 3: 修改配置并运行

```bash
# 1. 编辑配置
vi predict_from_weights.py
# 修改 INFERENCE_MODE 和相关参数

# 2. 运行推理
python predict_from_weights.py

# 3. 查看输出
head predictions_2024.csv
```

---

## 配置示例

### 示例1: 使用GPU 0, 1, 2（推荐）

```python
# predict_from_weights.py 中的配置
INFERENCE_MODE = "multi_gpu"
INFERENCE_GPU = 0
INFERENCE_GPUS = [0, 1, 2]
```

### 示例2: 只使用CPU

```python
INFERENCE_MODE = "cpu"
INFERENCE_GPU = -1
INFERENCE_GPUS = None
```

### 示例3: 使用GPU 1 和 2（如果GPU 0被占用）

```python
INFERENCE_MODE = "multi_gpu"
INFERENCE_GPU = 1  # 主GPU
INFERENCE_GPUS = [1, 2]  # 使用的GPU列表
```

---

## 性能对比（预估）

| 推理模式 | 显存占用 | 速度 | 适用场景 |
|---------|---------|------|---------|
| **多GPU (3卡)** | ~4 GB/卡 | **最快** (1-3分钟) | ⭐ 推荐，日常使用 |
| **单GPU** | ~12 GB | 中等 (3-5分钟) | 显存充足时 |
| **CPU** | 0 GB | 较慢 (10-30分钟) | GPU被占用时 |

---

## 验证配置是否生效

运行脚本后，查看输出中的这几行：

```
GPU Configuration:
  Mode: Multi-GPU      <-- 确认模式
  GPUs: [0, 1, 2]     <-- 确认GPU列表

✓ Model moved to: cuda:0
✓ Model wrapped with DataParallel on GPUs: [0, 1, 2]  <-- 确认DataParallel包装
✓ Model set to evaluation mode
```

---

## 常见问题

### Q1: 运行时还是显示单GPU

**问题**：配置了多GPU，但还是用单GPU

**检查**：
```python
# 确认这两行都设置了
model_config['GPU'] = INFERENCE_GPU
model_config['gpus'] = INFERENCE_GPUS
```

### Q2: CPU模式太慢

**优化**：
```python
# 增加CPU线程数
model_config['n_jobs'] = 20  # 根据CPU核心数调整

# 减少数据量（测试用）
ORIGINAL_TEST_SEGMENT = ("2024-01-01", "2024-01-31")  # 只预测1个月
```

### Q3: 多GPU推理还是OOM

**可能原因**：
- GPU被其他进程占用
- batch size太大

**解决**：
```bash
# 1. 清空GPU缓存
python -c "import torch; torch.cuda.empty_cache()"

# 2. 检查GPU占用
nvidia-smi

# 3. 杀死其他占用GPU的进程
kill <PID>
```

---

## 监控推理进度

### 查看GPU使用情况

```bash
# 实时监控GPU
watch -n 1 nvidia-smi

# 或者
nvidia-smi dmon -s u
```

### 查看进程

```bash
# 查看Python进程
ps aux | grep python

# 查看推理脚本运行状态
tail -f nohup.out  # 如果使用nohup运行
```

---

## 总结

**推荐配置**（针对你的情况）：

```python
# predict_from_weights.py

# 使用3张GPU推理（与训练一致）
INFERENCE_MODE = "multi_gpu"
INFERENCE_GPU = 0
INFERENCE_GPUS = [0, 1, 2]

# 如果GPU被占用，临时使用CPU
# INFERENCE_MODE = "cpu"
# INFERENCE_GPU = -1
# INFERENCE_GPUS = None
```

运行：
```bash
python predict_from_weights.py
```

预计时间：
- 多GPU: **1-3 分钟**
- CPU: 10-30 分钟

生成文件：
- `predictions_2024.csv`
- `predictions_2025.csv`
