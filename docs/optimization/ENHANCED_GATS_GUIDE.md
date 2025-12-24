# Enhanced GAT_ts 完整使用指南

## 🎯 新功能概览

| 功能 | 说明 | 使用方式 |
|------|------|----------|
| **最佳Checkpoint保存** | 自动保存训练中的最佳模型 | 自动，无需配置 |
| **Ctrl+C优雅中断** | 按Ctrl+C后自动保存最佳模型 | 训练时按Ctrl+C |
| **增强训练日志** | 详细的训练进度和指标显示 | 自动输出 |
| **MLflow集成** | 实验跟踪和可视化 | `use_mlflow=True` |
| **定期Checkpoint** | 每N个epoch保存一次 | `save_checkpoint_interval=5` |
| **从Checkpoint恢复** | 恢复训练或用于推理 | `model.load_checkpoint()` |

---

## 🚀 快速开始

### 1. 基础训练

```python
from qlib.contrib.model.pytorch_gats_ts_enhanced import GATs

model = GATs(
    d_feat=6,
    hidden_size=64,
    num_layers=2,
    dropout=0.0,
    n_epochs=200,
    lr=0.001,
    # 新增的参数
    checkpoint_dir="./checkpoints/my_model",  # checkpoint保存目录
    save_checkpoint_interval=5,  # 每5个epoch保存一次
    use_mlflow=True,  # 启用MLflow记录
)

# 训练（支持Ctrl+C中断）
model.fit(dataset, save_path="best_model.pth")
```

### 2. 从Checkpoint恢复

```python
# 创建模型
model = GATs(d_feat=6, hidden_size=64, num_layers=2)

# 加载checkpoint
checkpoint = model.load_checkpoint("./checkpoints/my_model/best_checkpoint.pth")

# 设置为已训练状态
model.fitted = True

# 直接预测
pred = model.predict(dataset)
```

### 3. 使用示例脚本

```bash
# 完整训练流程（含回测）
python train_gats_enhanced_example.py

# 从checkpoint恢复并回测
python train_gats_enhanced_example.py resume
```

---

## 📝 训练日志详解

### 增强前（原版）
```
Epoch0:
training...
evaluating...
train -0.001234, valid -0.001456
```

### 增强后
```
============================================================
Epoch 1/200 | Time: 45.32s
============================================================
  Train Loss: 0.001234 | Train Score: -0.001234
  Valid Loss: 0.001456 | Valid Score: -0.001456
  Best Score: -0.001456 @ Epoch 1
  Stop Steps: 0/20
  ★ New best model!
  ✓ Checkpoint saved: ./checkpoints/best_checkpoint.pth
```

**包含信息**：
- ⏱️ 每个epoch的训练时间
- 📊 训练loss和score
- 📊 验证loss和score
- 🏆 最佳score和对应epoch
- ⏸️ Early stopping进度
- 💾 Checkpoint保存提示

---

## 💾 Checkpoint管理

### Checkpoint文件结构

```
./checkpoints/my_model/
├── best_checkpoint.pth           # 最佳模型（自动保存）
├── checkpoint_epoch_5.pth        # 第5个epoch（定期保存）
├── checkpoint_epoch_10.pth       # 第10个epoch
├── checkpoint_epoch_15.pth       # 第15个epoch
└── ...
```

### Checkpoint内容

```python
checkpoint = torch.load("best_checkpoint.pth")

# 包含以下信息：
{
    'epoch': 42,                      # 保存时的epoch
    'model_state_dict': {...},        # 模型权重
    'optimizer_state_dict': {...},    # 优化器状态
    'score': -0.001234,              # 当时的验证score
    'best_score': -0.001234,         # 历史最佳score
    'best_epoch': 42,                # 最佳epoch
}
```

### 管理Checkpoint

```bash
# 查看所有checkpoint
ls -lh ./checkpoints/my_model/

# 删除旧的checkpoint（保留最佳的）
rm ./checkpoints/my_model/checkpoint_epoch_*.pth

# 备份最佳checkpoint
cp ./checkpoints/my_model/best_checkpoint.pth ./backups/
```

---

## ⌨️ Ctrl+C 中断处理

### 工作流程

```
训练中... → 按Ctrl+C → 检测到中断信号 → 保存最佳checkpoint → 优雅退出
```

### 实际效果

```bash
$ python train_gats_enhanced_example.py

============================================================
Epoch 42/200 | Time: 45.32s
============================================================
  Train Loss: 0.001234 | Train Score: -0.001234
  Valid Loss: 0.001456 | Valid Score: -0.001456
  Best Score: -0.001234 @ Epoch 38
  Stop Steps: 4/20

# 此时按 Ctrl+C

============================================================
Ctrl+C detected! Saving best checkpoint...
============================================================

============================================================
Training interrupted by user
============================================================

============================================================
Training Summary
============================================================
  Total Time: 31.50 minutes
  Best Score: -0.001234 @ Epoch 38
  Status: Interrupted
============================================================
Restoring best model...
  ✓ Best model saved to: best_model.pth
```

### 中断后继续回测

```python
# 方式1: 使用保存的最终模型
model = GATs(...)
model.fitted = True  # 标记为已训练
pred = model.predict(dataset)

# 方式2: 从checkpoint加载
model = GATs(...)
model.load_checkpoint("./checkpoints/best_checkpoint.pth")
model.fitted = True
pred = model.predict(dataset)
```

---

## 📊 MLflow集成

### 自动记录的内容

#### 1. 参数（Parameters）
```python
{
    "d_feat": 6,
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.0,
    "lr": 0.001,
    "optimizer": "adam",
    "base_model": "LSTM",
    "n_epochs": 200,
    "early_stop": 20,
}
```

#### 2. 指标（Metrics）- 每个epoch
```python
{
    "train_loss": 0.001234,
    "train_score": -0.001234,
    "val_loss": 0.001456,
    "val_score": -0.001456,
    "epoch_time": 45.32,
}
```

#### 3. 最终指标
```python
{
    "best_score": -0.001234,
    "best_epoch": 38,
    "total_training_time": 1890.5,
}
```

#### 4. 文件（Artifacts）
- 最佳模型文件
- 训练配置
- 预测结果

### 查看MLflow UI

```bash
# 启动UI
mlflow ui

# 浏览器打开
http://localhost:5000
```

### UI功能
- 📈 **图表**: 训练曲线可视化
- 📊 **对比**: 多个实验并排对比
- 📁 **文件**: 下载模型和结果
- 🔍 **搜索**: 筛选特定实验

---

## 🎓 完整使用示例

### 场景1: 正常训练流程

```python
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.contrib.model.pytorch_gats_ts_enhanced import GATs

# 初始化
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

# 创建数据集
dataset_config = {...}
dataset = init_instance_by_config(dataset_config)

# 创建模型
model = GATs(
    d_feat=6,
    hidden_size=64,
    num_layers=2,
    checkpoint_dir="./checkpoints/exp1",
    save_checkpoint_interval=5,
    use_mlflow=True,
)

# 在MLflow中记录
with R.start(experiment_name="gats_baseline"):
    # 训练
    model.fit(dataset, save_path="gats_baseline.pth")

    # 预测
    pred = model.predict(dataset)

    # 保存结果
    R.save_objects(**{"pred.pkl": pred})
```

### 场景2: 训练被中断后恢复回测

```bash
# 1. 训练时按了Ctrl+C
$ python train_gats_enhanced_example.py
# ... 训练中 ...
# 按 Ctrl+C
# ✓ 最佳模型已保存

# 2. 从checkpoint恢复并回测
$ python train_gats_enhanced_example.py resume
# 加载checkpoint...
# 开始预测...
# 开始回测...
# ✓ 完成！
```

### 场景3: 对比多个实验

```python
# 实验1: 学习率0.001
with R.start(experiment_name="lr_comparison"):
    R.log_params({"lr": 0.001})
    model = GATs(lr=0.001, checkpoint_dir="./checkpoints/lr_001")
    model.fit(dataset)

# 实验2: 学习率0.0001
with R.start(experiment_name="lr_comparison"):
    R.log_params({"lr": 0.0001})
    model = GATs(lr=0.0001, checkpoint_dir="./checkpoints/lr_0001")
    model.fit(dataset)

# 在MLflow UI中对比两个实验
```

---

## 🔧 高级配置

### 自定义Checkpoint策略

```python
model = GATs(
    ...
    # 每2个epoch保存一次
    save_checkpoint_interval=2,

    # 指定checkpoint目录
    checkpoint_dir="./my_checkpoints",
)
```

### 禁用某些功能

```python
model = GATs(
    ...
    # 禁用MLflow
    use_mlflow=False,

    # 禁用定期checkpoint（只保存最佳）
    save_checkpoint_interval=999999,
)
```

### 手动保存Checkpoint

```python
# 在训练循环中
for epoch in range(n_epochs):
    # ... 训练代码 ...

    # 手动保存
    if epoch % 10 == 0:
        model.save_checkpoint(
            epoch=epoch,
            score=val_score,
            is_best=False,
            filename=f"manual_checkpoint_epoch_{epoch}.pth"
        )
```

---

## 📈 性能对比

### 原版 vs 增强版

| 特性 | 原版 | 增强版 |
|------|------|--------|
| **日志信息** | 简单（1行） | 详细（多行+格式化） |
| **Checkpoint** | 仅最终模型 | 最佳+定期保存 |
| **中断处理** | 训练丢失 | 自动保存最佳 |
| **实验跟踪** | 手动记录 | MLflow自动 |
| **可视化** | 无 | MLflow UI |
| **训练时间** | 基准 | +1% (开销可忽略) |

---

## 🐛 故障排除

### Q1: 找不到checkpoint文件

**A**: 检查checkpoint_dir路径
```python
from pathlib import Path
checkpoint_dir = Path("./checkpoints/my_model")
print(f"Checkpoint目录: {checkpoint_dir.absolute()}")
print(f"是否存在: {checkpoint_dir.exists()}")
print(f"文件列表: {list(checkpoint_dir.glob('*.pth'))}")
```

### Q2: MLflow UI显示空白

**A**: 确保在正确的目录启动
```bash
# 在项目根目录（包含mlruns目录）启动
cd /path/to/project
ls mlruns  # 应该能看到实验文件夹
mlflow ui
```

### Q3: Ctrl+C后模型没保存

**A**: 检查信号处理
```python
# 确保没有自定义信号处理器覆盖
import signal
print(signal.getsignal(signal.SIGINT))  # 应该显示GATs的处理器
```

### Q4: 加载checkpoint报错

**A**: 确保模型配置一致
```python
# 保存时的配置
model_save = GATs(d_feat=6, hidden_size=64, num_layers=2)

# 加载时必须使用相同配置
model_load = GATs(d_feat=6, hidden_size=64, num_layers=2)
model_load.load_checkpoint("checkpoint.pth")
```

---

## 📚 相关文档

- [MLFLOW_GUIDE.md](MLFLOW_GUIDE.md) - MLflow详细使用指南
- [MODEL_USAGE_GUIDE.md](MODEL_USAGE_GUIDE.md) - 模型加载和推理指南
- [README_SCRIPTS.md](README_SCRIPTS.md) - 所有脚本说明

---

## ✅ 最佳实践

### 1. 目录组织

```
project/
├── checkpoints/           # 所有checkpoint
│   ├── exp1/
│   ├── exp2/
│   └── exp3/
├── mlruns/               # MLflow数据
├── models/               # 最终模型
│   ├── baseline.pth
│   └── best_model.pth
└── logs/                 # 训练日志
```

### 2. 命名规范

```python
# Checkpoint目录命名
checkpoint_dir = f"./checkpoints/gats_{base_model}_lr{lr}_h{hidden_size}"

# 模型文件命名
save_path = f"models/gats_{base_model}_{dataset_name}_best.pth"

# 实验命名
experiment_name = f"gats_{base_model}_{dataset_name}_{timestamp}"
```

### 3. 定期清理

```bash
# 定期删除旧的非最佳checkpoint
find ./checkpoints -name "checkpoint_epoch_*.pth" -mtime +7 -delete

# 备份重要checkpoint
tar -czf checkpoints_backup_$(date +%Y%m%d).tar.gz ./checkpoints/*/best_checkpoint.pth
```

---

## 🎉 总结

### 增强版的优势

1. **更安全**: Ctrl+C不会丢失训练成果
2. **更直观**: 详细的训练日志
3. **更专业**: MLflow实验管理
4. **更灵活**: 支持从任意checkpoint恢复
5. **更高效**: 自动化程度高

### 推荐使用场景

- ✅ **长时间训练** - 支持中断恢复
- ✅ **调参实验** - MLflow记录对比
- ✅ **团队协作** - 共享实验结果
- ✅ **生产环境** - Checkpoint管理
- ✅ **研究论文** - 完整实验追踪

---

**开始使用**: `python train_gats_enhanced_example.py`
