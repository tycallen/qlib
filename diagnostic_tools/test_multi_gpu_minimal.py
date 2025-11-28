#!/usr/bin/env python
"""
最小化测试：定位multi-GPU segfault
"""
import torch
import torch.nn as nn

print("="*60)
print("开始测试...")
print("="*60)

# 配置
device = torch.device("cuda:0")
gpus = [0, 1, 2]

# 测试1: 基础LSTM
print("\n[测试1] 基础LSTM (单GPU)")
try:
    lstm = nn.LSTM(20, 64, num_layers=2, batch_first=True, dropout=0.7)
    lstm = lstm.to(device)
    x = torch.randn(100, 20, 20).to(device)
    out, _ = lstm(x)
    print(f"✓ 通过")
except Exception as e:
    print(f"✗ 失败: {e}")
    import sys
    sys.exit(1)

# 测试2: DataParallel推理
print("\n[测试2] DataParallel推理")
try:
    lstm_dp = nn.DataParallel(lstm, device_ids=gpus)
    out, _ = lstm_dp(x)
    print(f"✓ 通过")
except Exception as e:
    print(f"✗ 失败: {e}")
    import traceback
    traceback.print_exc()
    import sys
    sys.exit(1)

# 测试3: DataParallel单步训练
print("\n[测试3] DataParallel单步训练")
try:
    optimizer = torch.optim.Adam(lstm_dp.parameters(), lr=0.001)
    out, _ = lstm_dp(x)
    loss = out.sum()
   
    print(f"  前向传播完成, loss={loss.item():.4f}")
    
    optimizer.zero_grad()
    print(f"  zero_grad完成")
    
    loss.backward()
    print(f"  反向传播完成")
    
    torch.nn.utils.clip_grad_value_(lstm_dp.parameters(), 3.0)
    print(f"  梯度裁剪完成")
    
    optimizer.step()
    print(f"✓ 完整训练步骤通过")
except Exception as e:
    print(f"✗ 失败在: {e}")
    import traceback
    traceback.print_exc()
    import sys
    sys.exit(1)

# 测试4: 多步训练循环
print("\n[测试4] 多步训练循环")
try:
    for i in range(5):
        x = torch.randn(100, 20, 20).to(device)
        out, _ = lstm_dp(x)
        loss = out.sum()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(lstm_dp.parameters(), 3.0)
        optimizer.step()
        
        print(f"  步骤 {i+1}/5: loss={loss.item():.4f}")
    print(f"✓ 多步训练通过")
except Exception as e:
    print(f"✗ 失败at步骤{i+1}: {e}")
    import traceback
    traceback.print_exc()
    import sys
    sys.exit(1)

print("\n" + "="*60)
print("所有测试通过！环境支持DataParallel+LSTM")
print("="*60)
print("\n如果此测试通过但实际训练失败，问题可能在:")
print("1. DataLoader (尝试 n_jobs=0)")
print("2. 数据预处理")
print("3. 特定的数据batch大小/形状")
