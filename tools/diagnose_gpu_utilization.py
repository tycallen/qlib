"""
诊断GPU利用率低的问题

常见原因：
1. 数据加载瓶颈 - CPU准备数据太慢
2. Batch size太小 - GPU吃不饱
3. 频繁CPU-GPU传输 - Daily batch导致
4. num_workers不足 - 数据加载线程太少
"""

import time
import torch
import numpy as np
from pathlib import Path


def diagnose_dataloader(data_loader, device, model=None):
    """诊断DataLoader性能"""

    print("="*60)
    print("DataLoader性能诊断")
    print("="*60)

    # 统计信息
    batch_sizes = []
    load_times = []
    transfer_times = []
    forward_times = []

    print("\n正在采样前10个batch...")

    for i, data in enumerate(data_loader):
        if i >= 10:  # 只测试前10个batch
            break

        # 1. 记录batch size
        if isinstance(data, torch.Tensor):
            batch_size = len(data)
        elif isinstance(data, (tuple, list)):
            batch_size = len(data[0])
        else:
            batch_size = -1
        batch_sizes.append(batch_size)

        # 2. 测试数据传输时间
        start = time.time()
        if isinstance(data, torch.Tensor):
            data = data.to(device)
        elif isinstance(data, (tuple, list)):
            data = tuple(d.to(device) if isinstance(d, torch.Tensor) else d for d in data)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        transfer_time = time.time() - start
        transfer_times.append(transfer_time)

        # 3. 测试前向传播时间（如果提供了模型）
        if model is not None:
            start = time.time()
            with torch.no_grad():
                try:
                    data_tensor = data.squeeze() if isinstance(data, torch.Tensor) else data[0].squeeze()
                    feature = data_tensor[:, :, 0:-1]
                    _ = model(feature.float())
                except:
                    pass
            torch.cuda.synchronize() if device.type == 'cuda' else None
            forward_time = time.time() - start
            forward_times.append(forward_time)

    # 输出诊断结果
    print("\n" + "="*60)
    print("诊断结果")
    print("="*60)

    print(f"\n📊 Batch统计:")
    print(f"  平均batch size: {np.mean(batch_sizes):.1f}")
    print(f"  最小batch size: {min(batch_sizes)}")
    print(f"  最大batch size: {max(batch_sizes)}")
    print(f"  Batch size标准差: {np.std(batch_sizes):.1f}")

    if min(batch_sizes) < 100:
        print(f"  ⚠️  警告: 最小batch size太小 ({min(batch_sizes)})，GPU利用率会很低！")

    if np.std(batch_sizes) > 100:
        print(f"  ⚠️  警告: Batch size波动大，训练效率不稳定！")

    print(f"\n⏱️  数据传输时间:")
    print(f"  平均: {np.mean(transfer_times)*1000:.2f} ms")
    print(f"  最大: {max(transfer_times)*1000:.2f} ms")

    if np.mean(transfer_times) > 0.1:
        print(f"  ⚠️  警告: 数据传输时间过长，建议使用pin_memory=True！")

    if model is not None and forward_times:
        print(f"\n🚀 前向传播时间:")
        print(f"  平均: {np.mean(forward_times)*1000:.2f} ms")
        print(f"  最大: {max(forward_times)*1000:.2f} ms")

        # 计算GPU利用率
        total_time = np.mean(transfer_times) + np.mean(forward_times)
        compute_ratio = np.mean(forward_times) / total_time if total_time > 0 else 0

        print(f"\n📈 计算效率:")
        print(f"  计算占比: {compute_ratio*100:.1f}%")
        print(f"  传输占比: {(1-compute_ratio)*100:.1f}%")

        if compute_ratio < 0.5:
            print(f"  ⚠️  警告: 计算时间占比过低，GPU大部分时间在等待数据！")

    # 给出建议
    print("\n" + "="*60)
    print("优化建议")
    print("="*60)

    suggestions = []

    if min(batch_sizes) < 100:
        suggestions.append("1. Batch size太小 → 增大batch size或改用固定batch")

    if np.mean(transfer_times) > 0.1:
        suggestions.append("2. 数据传输慢 → 使用pin_memory=True")

    suggestions.append("3. 增加num_workers（建议4-8倍CPU核心数）")
    suggestions.append("4. 使用persistent_workers=True（PyTorch >= 1.7）")
    suggestions.append("5. 考虑数据预处理和缓存")

    if model is not None and forward_times and compute_ratio < 0.5:
        suggestions.append("6. 使用预取（prefetch）加速数据加载")

    for suggestion in suggestions:
        print(f"  • {suggestion}")


def profile_training_step(train_loader, model, device, optimizer):
    """性能剖析：训练一个epoch"""

    print("\n" + "="*60)
    print("训练性能剖析")
    print("="*60)

    model.train()

    data_load_times = []
    forward_times = []
    backward_times = []

    print("\n测试前5个batch...")

    last_time = time.time()

    for i, data in enumerate(train_loader):
        if i >= 5:
            break

        # 1. 数据加载时间
        data_load_time = time.time() - last_time
        data_load_times.append(data_load_time)

        # 2. 数据传输 + 前向传播
        start = time.time()
        data = data.squeeze()
        feature = data[:, :, 0:-1].to(device)
        label = data[:, -1, -1].to(device)

        pred = model(feature.float())
        loss = ((pred - label) ** 2).mean()

        torch.cuda.synchronize() if device.type == 'cuda' else None
        forward_time = time.time() - start
        forward_times.append(forward_time)

        # 3. 反向传播
        start = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize() if device.type == 'cuda' else None
        backward_time = time.time() - start
        backward_times.append(backward_time)

        last_time = time.time()

    # 输出结果
    print("\n时间分布:")
    print(f"  数据加载: {np.mean(data_load_times)*1000:.2f} ms ({np.mean(data_load_times)/(np.mean(data_load_times)+np.mean(forward_times)+np.mean(backward_times))*100:.1f}%)")
    print(f"  前向传播: {np.mean(forward_times)*1000:.2f} ms ({np.mean(forward_times)/(np.mean(data_load_times)+np.mean(forward_times)+np.mean(backward_times))*100:.1f}%)")
    print(f"  反向传播: {np.mean(backward_times)*1000:.2f} ms ({np.mean(backward_times)/(np.mean(data_load_times)+np.mean(forward_times)+np.mean(backward_times))*100:.1f}%)")

    total_time = np.mean(data_load_times) + np.mean(forward_times) + np.mean(backward_times)
    gpu_time = np.mean(forward_times) + np.mean(backward_times)
    gpu_util = gpu_time / total_time * 100

    print(f"\n估算GPU利用率: {gpu_util:.1f}%")

    if np.mean(data_load_times) > gpu_time:
        print("\n⚠️  严重瓶颈: 数据加载时间 > GPU计算时间！")
        print("GPU大部分时间都在等待数据，这就是利用率低的主要原因！")


def check_system_info():
    """检查系统信息"""

    print("="*60)
    print("系统信息")
    print("="*60)

    print(f"\n🔧 PyTorch版本: {torch.__version__}")
    print(f"🐍 CUDA可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"🎮 GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"   显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")

    import multiprocessing
    print(f"💻 CPU核心数: {multiprocessing.cpu_count()}")


if __name__ == "__main__":
    print("GPU利用率诊断工具")
    print("="*60)
    print("\n使用方法:")
    print("在训练脚本中添加以下代码:\n")
    print("""
from diagnose_gpu_utilization import diagnose_dataloader, profile_training_step

# 在创建dataloader后
diagnose_dataloader(train_loader, device, model.GAT_model)

# 在训练前
profile_training_step(train_loader, model.GAT_model, device, optimizer)
""")

    check_system_info()
