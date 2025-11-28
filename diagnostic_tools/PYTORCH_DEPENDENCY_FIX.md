# PyTorch依赖冲突修复指南

## 🚨 当前问题

```
ImportError: undefined symbol: iJIT_NotifyEvent
```

这是PyTorch降级时MKL库版本冲突导致的。

## ✅ 快速修复（3个方案）

### 方案1: 完全重装（推荐）⭐

```bash
# 1. 完全卸载
conda uninstall pytorch torchvision torchaudio pytorch-cuda mkl mkl-service -y

# 2. 清理缓存
conda clean --all -y

# 3. 重装PyTorch 2.0.1
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 4. 验证
python -c "import torch; print(torch.__version__)"
```

### 方案2: 换CUDA版本

如果方案1失败：

```bash
conda uninstall pytorch torchvision torchaudio pytorch-cuda -y
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y
```

### 方案3: 使用pip安装

如果conda持续失败：

```bash
# 卸载conda版本
conda uninstall pytorch torchvision torchaudio -y

# 用pip安装
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

### 方案4: 重建conda环境（最彻底）

如果以上都失败：

```bash
# 1. 创建全新环境
conda create -n qlib_new python=3.10 -y
conda activate qlib_new

# 2. 安装PyTorch
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 3. 安装qlib依赖
pip install pyqlib

# 4. 测试
python -c "import torch; print(torch.__version__)"
```

## 🎯 自动化脚本

运行自动修复脚本：

```bash
cd /data10/tyc/stock/qlib_own
chmod +x fix_pytorch_env.sh
./fix_pytorch_env.sh
```

## 🔍 验证步骤

修复后验证：

```bash
# 1. 检查PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 2. 检查GPU
python -c "import torch; print(f'GPU数量: {torch.cuda.device_count()}')"

# 3. 测试多GPU
python test_multi_gpu_minimal.py
```

## 💡 如果仍然失败

### 检查系统CUDA

```bash
# 检查CUDA版本
nvidia-smi
nvcc --version

# 检查库路径
echo $LD_LIBRARY_PATH
```

### 尝试不同PyTorch版本

```bash
# PyTorch 2.1.0
conda install pytorch==2.1.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# PyTorch 1.13.1 (更稳定)
conda install pytorch==1.13.1 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

## 🆘 临时解决方案

如果紧急需要训练，使用单GPU：

```yaml
# workflow_config中改为单GPU
model:
    kwargs:
        gpus: [0]
        use_amp: true
```

然后：
```bash
# 回退到PyTorch 2.9.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia

# 单GPU训练
qrun examples/benchmarks/GATs/workflow_config_gats_Alpha158.yaml
```

## 📝 推荐版本组合

| PyTorch | CUDA | 稳定性 | 推荐 |
|---------|------|--------|------|
| 2.0.1 | 11.8 | ⭐⭐⭐⭐⭐ | ✅ 最推荐 |
| 2.1.0 | 11.8 | ⭐⭐⭐⭐ | ✅ 备选 |
| 1.13.1 | 11.7 | ⭐⭐⭐⭐ | ✅ 保守 |
| 2.9.1 | 12.8 | ⭐ (多GPU bug) | ❌ 避免 |
