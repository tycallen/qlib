"""
快速配置推理模式

使用方法：
    python configure_inference.py --mode multi_gpu --gpus 0,1,2
    python configure_inference.py --mode cpu
    python configure_inference.py --mode single_gpu --gpu 0
"""

import argparse
import re
from pathlib import Path


def update_inference_config(mode, gpu=0, gpus=None):
    """
    更新 predict_from_weights.py 中的推理配置

    Args:
        mode: 推理模式 (cpu, single_gpu, multi_gpu)
        gpu: 单GPU模式使用的GPU编号
        gpus: 多GPU模式使用的GPU列表
    """
    script_path = Path(__file__).parent / "predict_from_weights.py"

    if not script_path.exists():
        print(f"❌ Error: {script_path} not found!")
        return False

    # 读取脚本内容
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 构建新的配置
    if mode == "cpu":
        new_config = f'''    # 3. 推理模式配置（重要！）
    # 选项1: CPU模式（最安全，但较慢）
    INFERENCE_MODE = "cpu"
    INFERENCE_GPU = -1
    INFERENCE_GPUS = None'''
        print("\n✓ Configuring for CPU inference")
        print("  Device: CPU")
        print("  Note: Slower but no GPU memory required")

    elif mode == "single_gpu":
        new_config = f'''    # 3. 推理模式配置（重要！）
    # 选项2: 单GPU模式（如果单张显卡内存足够）
    INFERENCE_MODE = "single_gpu"
    INFERENCE_GPU = {gpu}
    INFERENCE_GPUS = None'''
        print(f"\n✓ Configuring for single GPU inference")
        print(f"  Device: GPU {gpu}")
        print(f"  Warning: May encounter OOM if GPU memory < 12GB")

    elif mode == "multi_gpu":
        if gpus is None:
            gpus = [0, 1, 2]  # 默认使用3张卡
        gpus_str = str(gpus)
        new_config = f'''    # 3. 推理模式配置（重要！）
    # 选项3: 多GPU模式（推荐，与训练时一致）
    INFERENCE_MODE = "multi_gpu"
    INFERENCE_GPU = {gpus[0]}
    INFERENCE_GPUS = {gpus_str}  # 使用{len(gpus)}张显卡'''
        print(f"\n✓ Configuring for multi-GPU inference")
        print(f"  Devices: GPUs {gpus}")
        print(f"  Recommended for best performance")

    else:
        print(f"❌ Unknown mode: {mode}")
        return False

    # 添加被注释掉的其他选项
    new_config += '''

    # 选项1: CPU模式（最安全，但较慢）
    # INFERENCE_MODE = "cpu"
    # INFERENCE_GPU = -1
    # INFERENCE_GPUS = None

    # 选项2: 单GPU模式（如果单张显卡内存足够）
    # INFERENCE_MODE = "single_gpu"
    # INFERENCE_GPU = 0
    # INFERENCE_GPUS = None

    # 选项3: 多GPU模式（推荐，与训练时一致）
    # INFERENCE_MODE = "multi_gpu"
    # INFERENCE_GPU = 0
    # INFERENCE_GPUS = [0, 1, 2]  # 使用3张显卡'''

    # 查找并替换配置部分
    pattern = r'    # 3\. 推理模式配置.*?# INFERENCE_GPUS = \[0, 1, 2\]  # 使用3张显卡'
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, new_config, content, flags=re.DOTALL)
    else:
        print("❌ Error: Could not find inference configuration section")
        return False

    # 写回文件
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\n✓ Configuration updated in: {script_path}")
    return True


def check_gpu_availability():
    """检查GPU可用性"""
    try:
        import torch
        if not torch.cuda.is_available():
            print("\n⚠ Warning: CUDA is not available!")
            print("  Only CPU mode will work.")
            return False

        num_gpus = torch.cuda.device_count()
        print(f"\n✓ CUDA available: {num_gpus} GPU(s) detected")

        for i in range(num_gpus):
            name = torch.cuda.get_device_name(i)
            total_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {name} ({total_mem:.1f} GB)")

        return True
    except ImportError:
        print("\n⚠ Warning: PyTorch not found, cannot check GPU")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Configure inference mode for predict_from_weights.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use 3 GPUs (recommended)
  python configure_inference.py --mode multi_gpu --gpus 0,1,2

  # Use CPU only
  python configure_inference.py --mode cpu

  # Use single GPU
  python configure_inference.py --mode single_gpu --gpu 0

  # Use GPU 1 and 2 (if GPU 0 is busy)
  python configure_inference.py --mode multi_gpu --gpus 1,2
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['cpu', 'single_gpu', 'multi_gpu'],
        required=True,
        help='Inference mode'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU index for single_gpu mode (default: 0)'
    )
    parser.add_argument(
        '--gpus',
        type=str,
        default=None,
        help='Comma-separated GPU indices for multi_gpu mode (e.g., "0,1,2")'
    )
    parser.add_argument(
        '--check-gpu',
        action='store_true',
        help='Check GPU availability before configuring'
    )

    args = parser.parse_args()

    print("="*60)
    print("Inference Mode Configuration Tool")
    print("="*60)

    # 检查GPU可用性
    if args.check_gpu:
        check_gpu_availability()

    # 解析GPU列表
    gpus_list = None
    if args.mode == 'multi_gpu':
        if args.gpus:
            try:
                gpus_list = [int(x.strip()) for x in args.gpus.split(',')]
            except ValueError:
                print(f"❌ Error: Invalid GPU list format: {args.gpus}")
                print("   Use format like: 0,1,2")
                return
        else:
            gpus_list = [0, 1, 2]  # 默认使用3张卡
            print(f"  Using default GPUs: {gpus_list}")

    # 更新配置
    success = update_inference_config(args.mode, args.gpu, gpus_list)

    if success:
        print("\n" + "="*60)
        print("Next steps:")
        print("="*60)
        print("  1. Run inference:")
        print("     python predict_from_weights.py")
        print()
        print("  2. Monitor GPU usage (if using GPU):")
        print("     watch -n 1 nvidia-smi")
        print()
        print("  3. View results:")
        print("     head predictions_2024.csv")
        print("="*60)


if __name__ == "__main__":
    main()
