# MLflow 实验跟踪指南

## MLflow 简介

MLflow 是一个开源的机器学习生命周期管理平台，非常适合用于深度学习模型的实验跟踪。

### 🎯 MLflow 的核心优势

#### 1. 实验管理 ✅
- **自动记录**: 训练参数、指标、模型文件
- **版本控制**: 每次实验都有唯一ID，方便追溯
- **对比分析**: 可视化对比不同实验的结果

#### 2. 可视化界面 📊
- **图表展示**: 训练loss、验证score等指标的曲线
- **参数对比**: 并排对比多个实验的参数设置
- **模型管理**: 统一管理所有保存的模型

#### 3. 协作友好 🤝
- **团队共享**: 团队成员可以查看彼此的实验结果
- **可复现**: 记录完整的训练配置，便于复现结果
- **笔记功能**: 为实验添加注释和说明

---

## 与Qlib的集成

Qlib本身已经集成了MLflow，在 `qlib.workflow` 中可以直接使用。

### 基础使用

```python
from qlib.workflow import R

# 开始一个实验
with R.start(experiment_name="my_experiment"):
    # 记录参数
    R.log_params({
        "learning_rate": 0.001,
        "hidden_size": 64,
        "num_layers": 2,
    })

    # 训练模型
    model.fit(dataset)

    # 记录指标
    R.log_metrics({
        "train_loss": 0.123,
        "val_score": 0.456,
    })

    # 保存模型
    R.save_objects(**{"model.pkl": model})
```

### Enhanced GAT_ts 中的使用

增强版GAT_ts已经自动集成了MLflow记录：

```python
from qlib.contrib.model.pytorch_gats_ts_enhanced import GATs

model = GATs(
    d_feat=6,
    hidden_size=64,
    num_layers=2,
    use_mlflow=True,  # 启用MLflow记录
    checkpoint_dir="./checkpoints",
)

# 自动记录以下内容：
# - 所有模型参数（d_feat, hidden_size等）
# - 每个epoch的训练指标（train_loss, val_loss, train_score, val_score）
# - 训练时间
# - 最佳模型信息
# - 模型文件
```

---

## MLflow UI 使用

### 启动MLflow UI

```bash
# 在项目目录下运行
mlflow ui

# 或指定端口
mlflow ui --host 0.0.0.0 --port 5001 --backend-store-uri ./mlruns

```

然后在浏览器中访问: `http://localhost:5000`

### UI 界面功能

#### 1. 实验列表
- 查看所有实验
- 按时间、指标排序
- 搜索和筛选

#### 2. 实验详情
- 参数表格
- 指标图表
- 模型文件下载
- 运行日志

#### 3. 对比功能
- 选择多个实验进行对比
- 并行坐标图
- 散点图矩阵

---

## MLflow vs 传统日志

### 传统方式的问题 ❌

```python
# 传统方式
print(f"Epoch {epoch}, loss: {loss}, score: {score}")

# 问题：
# 1. 难以追溯历史实验
# 2. 无法可视化对比
# 3. 参数和结果分散在不同地方
# 4. 无法方便地分享给团队
```

### MLflow的优势 ✅

```python
# MLflow方式
mlflow.log_params({"lr": 0.001, "hidden_size": 64})
mlflow.log_metrics({"loss": loss, "score": score}, step=epoch)

# 优势：
# 1. 自动组织和存储
# 2. 可视化图表自动生成
# 3. 参数和结果关联在一起
# 4. Web界面方便分享和查看
```

---

## 实战示例

### 场景1: 对比不同学习率

```python
import qlib
from qlib.workflow import R
from qlib.contrib.model.pytorch_gats_ts_enhanced import GATs

qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")

# 测试不同学习率
for lr in [0.0001, 0.001, 0.01]:
    with R.start(experiment_name="lr_comparison"):
        R.log_params({"learning_rate": lr})

        model = GATs(lr=lr, use_mlflow=True)
        model.fit(dataset)

        # MLflow自动记录所有指标

# 在MLflow UI中对比3个实验的结果
```

### 场景2: 记录额外信息

```python
with R.start(experiment_name="my_experiment"):
    # 记录训练配置
    R.log_params({
        "model": "GATs",
        "dataset": "Alpha360",
        "data_range": "2008-2020",
    })

    # 训练
    model.fit(dataset)

    # 记录自定义指标
    R.log_metrics({
        "best_epoch": model.best_epoch,
        "total_params": count_parameters(model),
    })

    # 保存额外文件
    R.save_objects(**{
        "train_history.json": train_history,
        "config.yaml": config,
    })
```

### 场景3: 从UI恢复实验

```python
from qlib.workflow import R

# 加载特定实验
rec = R.get_recorder(recorder_id="<experiment_id>")

# 获取参数
params = rec.list_params()
print(params)

# 获取指标
metrics = rec.list_metrics()
print(metrics)

# 加载模型
model = rec.load_object("model.pkl")

# 获取预测结果
pred = rec.load_object("pred.pkl")
```

---

## 最佳实践

### 1. 实验命名规范

```python
# 好的命名
with R.start(experiment_name="gats_lstm_baseline"):
    ...

with R.start(experiment_name="gats_lstm_lr0.001_hidden128"):
    ...

with R.start(experiment_name="gats_gru_alpha360_2008-2020"):
    ...

# 避免
with R.start(experiment_name="test"):  # 太模糊
    ...
```

### 2. 记录关键信息

```python
with R.start(experiment_name="my_experiment"):
    # 记录完整的训练配置
    R.log_params({
        # 模型参数
        "d_feat": 6,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.0,
        # 训练参数
        "lr": 0.001,
        "n_epochs": 200,
        "batch_size": 2000,
        # 数据参数
        "dataset": "Alpha360",
        "train_period": "2008-2014",
        "valid_period": "2015-2016",
        "test_period": "2017-2020",
        # 环境信息
        "gpu": "Tesla V100",
        "cuda_version": "11.3",
    })
```

### 3. 定期清理

```bash
# 删除旧的实验
mlflow experiments delete --experiment-id <id>

# 或在UI中标记为deleted
```

---

## 常见问题

### Q1: MLflow数据存储在哪里？

**A**: 默认存储在 `./mlruns` 目录下

```
./mlruns/
├── 0/                          # 默认实验
├── 123456789/                  # 实验ID
│   ├── <run_id>/              # 运行ID
│   │   ├── artifacts/         # 模型文件、图表等
│   │   ├── metrics/           # 指标数据
│   │   ├── params/            # 参数
│   │   └── tags/              # 标签
│   └── meta.yaml
└── .trash/                     # 已删除的实验
```

### Q2: 如何在远程服务器上使用MLflow？

**A**: 设置tracking server

```bash
# 服务器上启动MLflow server
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000

# 客户端设置tracking URI
export MLFLOW_TRACKING_URI=http://your-server:5000

# 或在代码中设置
import mlflow
mlflow.set_tracking_uri("http://your-server:5000")
```

### Q3: 如何导出MLflow数据？

**A**: 使用MLflow CLI或API

```bash
# 导出单个运行
mlflow runs export --run-id <run_id> --output-dir ./export

# 导出整个实验
mlflow experiments export --experiment-id <exp_id> --output-dir ./export
```

### Q4: MLflow会影响训练速度吗？

**A**: 影响很小（< 1%）

- 记录参数和指标的开销可忽略
- 文件保存是异步的，不阻塞训练
- 可以通过减少记录频率进一步优化

### Q5: 如何禁用MLflow？

**A**: 在Enhanced GAT_ts中设置：

```python
model = GATs(
    ...
    use_mlflow=False,  # 禁用MLflow
)
```

---

## 与Enhanced GAT_ts的完整工作流

### 1. 训练阶段

```bash
# 启动训练
python train_gats_enhanced_example.py

# 训练过程中：
# - 自动记录每个epoch的指标到MLflow
# - 定期保存checkpoint到./checkpoints
# - 可以随时Ctrl+C中断
```

### 2. 查看结果

```bash
# 启动MLflow UI
mlflow ui

# 浏览器访问 http://localhost:5000
# 查看：
# - 训练曲线（loss, score）
# - 最佳epoch
# - 训练时间
# - 所有参数设置
```

### 3. 对比实验

```bash
# 在UI中选择多个实验
# 点击"Compare"
# 查看：
# - 参数差异表格
# - 指标对比图表
# - 并行坐标图
```

### 4. 恢复训练

```python
# 从checkpoint恢复
python train_gats_enhanced_example.py resume

# 或手动加载
model = GATs(...)
model.load_checkpoint("./checkpoints/best_checkpoint.pth")
pred = model.predict(dataset)
```

---

## 总结

### MLflow好用吗？ ⭐⭐⭐⭐⭐

**非常好用！** 强烈推荐使用，理由：

#### ✅ 优点
1. **零学习成本**: Qlib已集成，开箱即用
2. **自动化程度高**: 大部分信息自动记录
3. **可视化强大**: Web UI直观美观
4. **团队协作**: 方便分享和对比实验
5. **可追溯性**: 永久保存所有历史实验
6. **生产就绪**: 可以直接部署到生产环境

#### ⚠️ 注意事项
1. 需要额外磁盘空间存储实验数据
2. 长期使用建议定期清理旧实验
3. 多人使用建议搭建共享的tracking server

#### 🎯 推荐使用场景
- ✅ 调参实验（必须用）
- ✅ 模型对比（必须用）
- ✅ 团队协作（必须用）
- ✅ 论文实验（强烈推荐）
- ✅ 生产部署（推荐）
- ⚠️ 快速原型开发（可选）

---

## 参考资源

- [MLflow官方文档](https://mlflow.org/docs/latest/index.html)
- [Qlib MLflow集成](https://qlib.readthedocs.io/en/latest/component/recorder.html)
- [MLflow最佳实践](https://mlflow.org/docs/latest/tracking.html#best-practices)

---

**结论**: MLflow是深度学习实验管理的标准工具，与Qlib的集成非常好，**强烈推荐使用**！
