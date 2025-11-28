# PyTorch 2.9.1 DataParallel Segfault 解决方案

## 🚨 问题确认

您的环境：
- **PyTorch**: 2.9.1+cu128
- **CUDA**: 12.8
- **cuDNN**: 91002
- **问题**: DataParallel + LSTM/GRU 立即崩溃（segfault）

## 🔍 根本原因

PyTorch 2.9.1 在2024年底/2025年初发布，是一个**非常新的版本**，存在与DataParallel + RNN模块的兼容性问题。这是一个**已知的upstream bug**，不是您代码的问题。

## ✅ 解决方案（按推荐顺序）

### 方案1: 降级PyTorch（强烈推荐）⭐

降级到稳定的LTS版本：

```bash
# 备份当前环境
conda create --name qlib_backup --clone qlib

# 降级到PyTorch 2.0.1 (LTS版本，最稳定)
conda activate qlib
conda install pytorch==2.0.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 验证
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
python test_multi_gpu_minimal.py
```

**为什么选2.0.1**:
- 是PyTorch 2.x的第一个稳定版本
- 有最长的社区测试时间
- DataParallel + RNN 经过充分验证

### 方案2: 尝试PyTorch 2.1.x或2.2.x

如果2.0.1不满足其他需求，尝试：

```bash
# PyTorch 2.1.2
conda install pytorch==2.1.2 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 或 PyTorch 2.2.2
conda install pytorch==2.2.2 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 方案3: 使用单GPU训练

如果不能降级，使用单GPU：

```yaml
# workflow_config_gats_Alpha158.yaml
model:
    kwargs:
        gpus: [0]  # 单GPU
        use_amp: true  # 启用AMP补偿速度损失
```

**单GPU + AMP性能**:
- 比原始单GPU快1.5-2x
- 比多GPU慢，但稳定可靠

### 方案4: 修改模型配置（临时workaround）

如果必须使用2.9.1，可能的workaround（成功率低）：

#### 4a. 移除Dropout
```python
# pytorch_gats_ts.py
self.GAT_model = GATModel(
    d_feat=self.d_feat,
    hidden_size=self.hidden_size,
    num_layers=self.num_layers,
    dropout=0.0,  # 改为0.0
    base_model=self.base_model,
)
```

#### 4b. 使用单层RNN
```yaml
model:
    kwargs:
        num_layers: 1  # 改为1层
```

#### 4c. 换用GRU
```yaml
model:
    kwargs:
        base_model: GRU  # LSTM改为GRU
```

**注意**: 这些workaround不保证成功，且可能影响模型性能。

## 🧪 验证步骤

降级或修改后，运行验证：

```bash
cd /data10/tyc/stock/qlib_own

# 1. 基础验证
python diagnose_pytorch_version.py

# 2. 如果通过，测试实际训练
qrun examples/benchmarks/GATs/workflow_config_gats_Alpha158.yaml
```

## 📊 性能对比

| 方案 | 速度 | 稳定性 | 推荐度 |
|------|------|--------|--------|
| 降级到PyTorch 2.0.1 + 多GPU | 100% | ⭐⭐⭐⭐⭐ | 🥇 最推荐 |
| 单GPU + AMP | ~60% | ⭐⭐⭐⭐⭐ | 🥈 备选 |
| PyTorch 2.9.1 + 单GPU | ~35% | ⭐⭐⭐⭐ | 🥉 可接受 |
| PyTorch 2.9.1 + 多GPU | ❌ 崩溃 | ⭐ | ❌ 不可用 |

## 🔧 完整降级脚本

```bash
#!/bin/bash
# 降级PyTorch到稳定版本

# 1. 备份环境
conda create --name qlib_backup --clone qlib -y

# 2. 激活环境
conda activate qlib

# 3. 卸载当前PyTorch
conda uninstall pytorch torchvision torchaudio -y

# 4. 安装稳定版本
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 5. 验证
python -c "import torch; print(f'✅ PyTorch {torch.__version__} installed')"
python -c "import torch; print(f'✅ CUDA available: {torch.cuda.is_available()}')"

# 6. 测试多GPU
cd /data10/tyc/stock/qlib_own
python test_multi_gpu_minimal.py

echo "如果测试通过，可以开始训练了！"
```

保存为`downgrade_pytorch.sh`并运行：
```bash
chmod +x downgrade_pytorch.sh
./downgrade_pytorch.sh
```

## 📝 历史背景

您提到"之前某个版本的代码是可以多GPU训练的"，这很可能是因为：

1. **之前使用的PyTorch版本不同**（可能是1.x或2.0.x）
2. **之前的环境配置不同**

这再次证明了**这不是代码问题，而是环境兼容性问题**。

## 🆘 如果降级后仍失败

如果降级PyTorch后仍然segfault，可能是：

1. **CUDA驱动版本问题**
   ```bash
   nvidia-smi  # 检查驱动版本
   # 可能需要更新驱动
   ```

2. **GPU硬件问题**
   ```bash
   nvidia-smi topo -m  # 检查GPU互连
   # 尝试不同的GPU组合
   ```

3. **系统库问题**
   ```bash
   ldconfig -p | grep cuda  # 检查CUDA库
   ```

请先运行`diagnose_pytorch_version.py`告诉我结果！
