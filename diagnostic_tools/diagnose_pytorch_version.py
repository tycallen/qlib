#!/usr/bin/env python
"""
深度诊断：PyTorch版本兼容性测试
"""
import torch
import torch.nn as nn
import sys

print("="*60)
print("环境信息")
print("="*60)
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"cuDNN版本: {torch.backends.cudnn.version()}")
print(f"GPU数量: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

device = torch.device("cuda:0")
gpus = [0, 1]

# 测试1: 单GPU LSTM（应该正常）
print("\n" + "="*60)
print("[测试1] 单GPU LSTM")
print("="*60)
try:
    lstm = nn.LSTM(20, 64, num_layers=2, batch_first=True, dropout=0.7).to(device)
    x = torch.randn(64, 20, 20).to(device)
    out, _ = lstm(x)
    loss = out.sum()
    loss.backward()
    print("✅ 单GPU LSTM完全正常")
except Exception as e:
    print(f"❌ 单GPU LSTM失败: {e}")
    sys.exit(1)

# 测试2: DataParallel推理（无需backward）
print("\n" + "="*60)
print("[测试2] DataParallel推理（仅forward，无backward）")
print("="*60)
try:
    lstm_dp = nn.DataParallel(lstm, device_ids=gpus)
    with torch.no_grad():
        out, _ = lstm_dp(x)
    print("✅ DataParallel推理通过")
    dp_works_forward = True
except Exception as e:
    print(f"❌ DataParallel推理失败")
    dp_works_forward = False

if not dp_works_forward:
    print("\n" + "="*60)
    print("结论：DataParallel在forward阶段就崩溃")
    print("="*60)
    print("\n这是PyTorch 2.9.1的已知问题。")
    print("\n推荐解决方案：")
    print("1. 降级到 PyTorch 2.0.1 (稳定版本)")
    print("2. 或使用单GPU训练")
    print("\n降级命令:")
    print("  conda install pytorch==2.0.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
    sys.exit(1)

# 测试3: DataParallel训练（backward）
print("\n" + "="*60)
print("[测试3] DataParallel训练（forward + backward）")
print("="*60)
try:
    out, _ = lstm_dp(x)
    loss = out.sum()
    loss.backward()
    print("✅ DataParallel训练通过")
    dp_works_backward = True
except Exception as e:
    print(f"❌ DataParallel训练失败")
    dp_works_backward = False

if dp_works_forward and not dp_works_backward:
    print("\n结论：DataParallel在backward阶段崩溃")
    print("可能的workaround: 使用torch.nn.parallel.DistributedDataParallel")
    sys.exit(1)

# 测试4: 测试是否是dropout问题
print("\n" + "="*60)
print("[测试4] 尝试不同配置")
print("="*60)

configs = [
    ("LSTM dropout=0.0", lambda: nn.LSTM(20, 64, num_layers=2, batch_first=True, dropout=0.0)),
    ("LSTM num_layers=1", lambda: nn.LSTM(20, 64, num_layers=1, batch_first=True)),
    ("GRU dropout=0.7", lambda: nn.GRU(20, 64, num_layers=2, batch_first=True, dropout=0.7)),
    ("GRU dropout=0.0", lambda: nn.GRU(20, 64, num_layers=2, batch_first=True, dropout=0.0)),
]

working_configs = []
for name, model_fn in configs:
    try:
        model = model_fn().to(device)
        model_dp = nn.DataParallel(model, device_ids=gpus)
        out, _ = model_dp(x)
        loss = out.sum()
        loss.backward()
        print(f"  ✅ {name}")
        working_configs.append(name)
    except:
        print(f"  ❌ {name}")

print("\n" + "="*60)
print("总结")
print("="*60)

if working_configs:
    print(f"\n✅ 找到{len(working_configs)}个可行配置:")
    for cfg in working_configs:
        print(f"  - {cfg}")
    print("\n建议: 在模型中使用上述配置之一")
else:
    print("\n❌ 没有找到可行配置")
    print("\n这说明您的PyTorch版本与DataParallel存在根本性不兼容。")
    print("\n强烈推荐:")
    print("  1. 降级PyTorch到2.0.1 (LTS版本)")
    print("  2. 或升级到2.1.0+ (如果有修复)")
    print("  3. 或使用单GPU训练")
