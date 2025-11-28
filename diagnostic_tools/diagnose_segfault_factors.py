#!/usr/bin/env python
"""
Segfault 因子隔离测试
用于找出导致DataParallel崩溃的具体参数
"""
import torch
import torch.nn as nn
import sys

# 配置
device = torch.device("cuda:0")
gpus = [0, 1]  # 只用2个GPU测试，减少变量
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"CUDNN: {torch.backends.cudnn.version()}")
print("="*60)

def run_test(name, model_fn):
    print(f"\n[测试] {name}...", end="", flush=True)
    try:
        # 1. 创建模型
        model = model_fn().to(device)
        
        # 2. DataParallel包装
        model = nn.DataParallel(model, device_ids=gpus)
        
        # 3. 推理测试
        x = torch.randn(64, 20, 20).to(device)
        out, _ = model(x)
        
        # 4. 反向传播测试
        loss = out.sum()
        loss.backward()
        
        print(" ✅ 通过")
        return True
    except Exception as e:
        print(f" ❌ 失败: {e}")
        return False

# --- 测试用例 ---

# 1. 原始配置 (基准)
def test_baseline():
    return nn.LSTM(20, 64, num_layers=2, batch_first=True, dropout=0.7)
run_test("基准 (LSTM, dropout=0.7)", test_baseline)

# 2. 移除 Dropout
def test_no_dropout():
    return nn.LSTM(20, 64, num_layers=2, batch_first=True, dropout=0.0)
run_test("无Dropout (dropout=0.0)", test_no_dropout)

# 3. 使用 GRU
def test_gru():
    return nn.GRU(20, 64, num_layers=2, batch_first=True, dropout=0.7)
run_test("使用 GRU (dropout=0.7)", test_gru)

# 4. 使用 GRU 无 Dropout
def test_gru_no_dropout():
    return nn.GRU(20, 64, num_layers=2, batch_first=True, dropout=0.0)
run_test("使用 GRU + 无Dropout", test_gru_no_dropout)

# 5. 单层 (无dropout)
def test_single_layer():
    return nn.LSTM(20, 64, num_layers=1, batch_first=True, dropout=0.0)
run_test("单层 LSTM (num_layers=1)", test_single_layer)

# 6. Flatten Parameters
def test_flatten():
    class FlattenLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(20, 64, num_layers=2, batch_first=True, dropout=0.7)
        def forward(self, x):
            self.rnn.flatten_parameters() # 显式调用
            return self.rnn(x)
    return FlattenLSTM()
run_test("带 flatten_parameters()", test_flatten)

print("\n" + "="*60)
print("测试结束。请查看哪个组合通过了。")
