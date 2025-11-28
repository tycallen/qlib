# 多GPU Segmentation Fault 调试指南

## 🔍 快速诊断

首先运行诊断脚本：

```bash
cd /data10/tyc/stock/qlib_own
python diagnose_multi_gpu.py
```

这将测试您的环境是否支持DataParallel + LSTM。

## 🐛 Debug方法

### 方法1: GDB调试（推荐）

```bash
# 使用GDB运行
gdb --args python -c "import torch; import sys; sys.path.insert(0, '/data10/tyc/stock/qlib_own'); from qlib.contrib.model.pytorch_gats_ts import GATs"

# 在GDB中
(gdb) run
# 等待segfault
(gdb) bt  # 查看堆栈跟踪
(gdb) info threads  # 查看线程信息
```

### 方法2: 启用PyTorch Debug模式

在配置文件顶部添加：

```python
import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'  # 启用CUDA同步调试
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步CUDA调用
```

然后运行：
```bash
qrun examples/benchmarks/GATs/workflow_config_gats_Alpha158.yaml
```

### 方法3: 逐步隔离问题

创建测试脚本 `test_minimal.py`:

```python
import torch
import torch.nn as nn

# 测试1: 单GPU
print("测试1: 单GPU...")
device = torch.device("cuda:0")
lstm = nn.LSTM(20, 64, num_layers=2, batch_first=True, dropout=0.7).to(device)
x = torch.randn(100, 20, 20).to(device)
out, _ = lstm(x)
print("✓ 单GPU通过")

# 测试2: DataParallel
print("\n测试2: DataParallel...")
lstm_dp = nn.DataParallel(lstm, device_ids=[0, 1, 2])
out, _ = lstm_dp(x)
print("✓ DataParallel推理通过")

# 测试3: DataParallel训练
print("\n测试3: DataParallel训练...")
optimizer = torch.optim.Adam(lstm_dp.parameters(), lr=0.001)
for i in range(3):
    out, _ = lstm_dp(x)
    loss = out.sum()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(lstm_dp.parameters(), 3.0)
    optimizer.step()
    print(f"  步骤 {i+1}: loss={loss.item():.4f}")
print("✓ DataParallel训练通过")
```

运行：
```bash
python test_minimal.py
```

## 🔧 常见问题和解决方案

### 问题1: CUDA版本不兼容

**症状**: Multi-GPU segfault，单GPU正常

**检查**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
nvidia-smi
```

**解决**: 确保PyTorch CUDA版本与系统CUDA版本兼容

### 问题2: DataLoader冲突

**症状**: 只在使用DataLoader时segfault

**检查**: `n_jobs` 参数

**解决**:
```yaml
model:
    kwargs:
        n_jobs: 0  # 必须为0！
```

### 问题3: GPU拓扑问题

**症状**: 特定GPU组合时segfault

**检查**:
```bash
nvidia-smi topo -m
```

**解决**: 使用P2P连接好的GPU组合，如:
```yaml
gpus: [0, 1]  # 而不是 [0, 2]
```

### 问题4: PyTorch版本问题

**已知Bug**: PyTorch < 1.10 在某些情况下DataParallel + LSTM有bug

**解决**: 升级到PyTorch >= 1.10

```bash
conda install pytorch>=1.10 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 问题5: Dropout与DataParallel

**症状**: 有dropout时segfault，无dropout正常

**临时解决**: 设置环境变量
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

## 🎯 逐步Troubleshooting流程

### Step 1: 确认单GPU可用
```bash
cd /data10/tyc/stock/qlib_own
# 修改配置为单GPU
vim examples/benchmarks/GATs/workflow_config_gats_Alpha158.yaml
# 改为: gpus: [0]

qrun examples/benchmarks/GATs/workflow_config_gats_Alpha158.yaml
```

如果单GPU失败→ 问题不在多GPU，检查模型代码

### Step 2: 测试最小DataParallel
```bash
python test_minimal.py
```

如果失败→ 环境问题（PyTorch/CUDA/Driver）

### Step 3: 逐步增加复杂度

1. 先用2个GPU: `gpus: [0, 1]`
2. 再用3个GPU: `gpus: [0, 1, 2]`
3. 加入真实数据

找到失败的临界点。

### Step 4: 检查训练vs推理

修改代码，在`train_epoch`前打印：
```python
print("[DEBUG] 开始forward...")
sys.stdout.flush()  # 强制输出
```

看segfault发生在哪个阶段。

## 💡 可能的根本原因

### 1. LSTM内部状态管理问题

DataParallel在多GPU间复制模型时，LSTM的hidden state可能出问题。

**测试**: 使用GRU替代LSTM
```yaml
model:
    kwargs:
        base_model: GRU  # 改为GRU测试
```

### 2. 梯度裁剪时机问题

`clip_grad_value_`在DataParallel可能有问题。

**测试**: 临时移除梯度裁剪
```python
# torch.nn.utils.clip_grad_value_(self.GAT_model.parameters(), 3.0)  # 注释掉
```

### 3. 数据类型问题

**测试**: 确保所有数据是float32
```python
feature = data[:, :, 0:-1].float().to(self.device)  # 显式转float32
```

## 📋 信息收集清单

运行后收集以下信息：

```bash
# 1. 环境信息
python diagnose_multi_gpu.py > debug_env.txt 2>&1

# 2. GPU拓扑
nvidia-smi topo -m > gpu_topology.txt

# 3. 详细错误
# 添加到脚本开头:
import faulthandler
faulthandler.enable()

# 4. CUDA错误
export CUDA_LAUNCH_BLOCKING=1
qrun ... 2>&1 | tee cuda_debug.log
```

## 🔄 回退方案

如果实在无法解决，考虑：

1. **使用单GPU** - 最稳定
2. **使用2个GPU** - 降低复杂度
3. **使用DistributedDataParallel** - 虽然内存大，但更稳定
4. **换用GRU替代LSTM** - 可能更稳定

## 📞 寻求帮助

如果以上都无效，提供：
1. `debug_env.txt`
2. `gpu_topology.txt`
3. `cuda_debug.log`
4. PyTorch/CUDA版本
5. GPU型号

这将帮助定位问题！
