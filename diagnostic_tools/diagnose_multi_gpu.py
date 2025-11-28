#!/usr/bin/env python
"""
多GPU Segmentation Fault 调试脚本

用于诊断DataParallel + LSTM/GRU的segfault问题
"""

import torch
import torch.nn as nn
import sys

print("=" * 60)
print("PyTorch 多GPU Segfault 诊断工具")
print("=" * 60)

# 1. 环境信息
print("\n[1] 环境信息:")
print(f"  PyTorch版本: {torch.__version__}")
print(f"  CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA版本: {torch.version.cuda}")
    print(f"  GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")

# 2. 测试单GPU LSTM
print("\n[2] 测试单GPU LSTM:")
try:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lstm = nn.LSTM(input_size=20, hidden_size=64, num_layers=2, batch_first=True, dropout=0.7)
    lstm = lstm.to(device)
    
    x = torch.randn(100, 20, 20).to(device)
    out, _ = lstm(x)
    print(f"  ✓ 单GPU LSTM测试通过 (输出形状: {out.shape})")
except Exception as e:
    print(f"  ✗ 单GPU LSTM测试失败: {e}")
    sys.exit(1)

# 3. 测试DataParallel LSTM (不带flatten_parameters)
print("\n[3] 测试DataParallel LSTM (无flatten_parameters):")
if torch.cuda.device_count() >= 2:
    try:
        gpus = list(range(min(3, torch.cuda.device_count())))
        print(f"  使用GPU: {gpus}")
        
        device = torch.device(f"cuda:{gpus[0]}")
        lstm = nn.LSTM(input_size=20, hidden_size=64, num_layers=2, batch_first=True, dropout=0.7)
        lstm = lstm.to(device)
        lstm = nn.DataParallel(lstm, device_ids=gpus)
        
        x = torch.randn(100, 20, 20).to(device)
        out, _ = lstm(x)
        print(f"  ✓ DataParallel LSTM测试通过 (输出形状: {out.shape})")
    except Exception as e:
        print(f"  ✗ DataParallel LSTM测试失败: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"  跳过 (需要至少2个GPU，当前只有{torch.cuda.device_count()}个)")

# 4. 测试DataParallel LSTM训练循环
print("\n[4] 测试DataParallel LSTM训练循环:")
if torch.cuda.device_count() >= 2:
    try:
        gpus = list(range(min(3, torch.cuda.device_count())))
        device = torch.device(f"cuda:{gpus[0]}")
        
        # 创建简单模型
        class SimpleLSTMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(20, 64, num_layers=2, batch_first=True, dropout=0.7)
                self.fc = nn.Linear(64, 1)
            
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :]).squeeze()
        
        model = SimpleLSTMModel().to(device)
        model = nn.DataParallel(model, device_ids=gpus)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 训练几步
        for i in range(5):
            x = torch.randn(100, 20, 20).to(device)
            y = torch.randn(100).to(device)
            
            pred = model(x)
            loss = nn.MSELoss()(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
            optimizer.step()
            
            print(f"    步骤 {i+1}/5: loss={loss.item():.4f}")
        
        print(f"  ✓ DataParallel训练循环测试通过")
    except Exception as e:
        print(f"  ✗ DataParallel训练循环测试失败: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"  跳过 (需要至少2个GPU)")

# 5. 检查潜在问题
print("\n[5] 潜在问题检查:")

# 检查PyTorch版本
torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
if torch_version < (1, 10):
    print(f"  ⚠️  PyTorch版本较旧 ({torch.__version__})，建议升级到1.10+")
else:
    print(f"  ✓ PyTorch版本合适 ({torch.__version__})")

# 检查CUDA版本兼容性
if torch.cuda.is_available():
    cuda_version = torch.version.cuda
    if cuda_version:
        cuda_major = int(cuda_version.split('.')[0])
        if cuda_major < 11:
            print(f"  ⚠️  CUDA版本较旧 ({cuda_version})，建议使用CUDA 11+")
        else:
            print(f"  ✓ CUDA版本合适 ({cuda_version})")

# 6. 建议
print("\n[6] 诊断建议:")
print("  如果以上测试失败，可能的原因包括:")
print("  1. PyTorch/CUDA版本不兼容")
print("  2. GPU驱动问题")
print("  3. DataLoader的num_workers与DataParallel冲突")
print("  4. LSTM的flatten_parameters()调用（应已移除）")
print("  5. 特定GPU硬件问题")
print("\n  建议尝试:")
print("  - 设置 n_jobs=0 (禁用DataLoader多进程)")
print("  - 减少GPU数量 (如只用2个GPU)")
print("  - 更新PyTorch和CUDA")
print("  - 检查GPU拓扑结构: nvidia-smi topo -m")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)
