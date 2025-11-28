#!/bin/bash
# PyTorch环境完全重建脚本

echo "=========================================="
echo "PyTorch环境修复方案"
echo "=========================================="

# 方案1: 完全卸载并重装PyTorch
echo -e "\n[方案1] 完全清理并重装PyTorch 2.0.1"
echo "----------------------------------------"
echo "1. 完全卸载PyTorch及相关包"
conda uninstall pytorch torchvision torchaudio pytorch-cuda mkl mkl-service -y

echo "2. 清理conda缓存"
conda clean --all -y

echo "3. 重新安装PyTorch 2.0.1"
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

echo "4. 验证安装"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 如果方案1失败，使用方案2
if [ $? -ne 0 ]; then
    echo -e "\n[方案2] 尝试其他CUDA版本"
    echo "----------------------------------------"
    conda uninstall pytorch torchvision torchaudio pytorch-cuda -y
    conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y
    python -c "import torch; print(f'PyTorch: {torch.__version__}')"
fi

# 如果还是失败，使用方案3
if [ $? -ne 0 ]; then
    echo -e "\n[方案3] 使用pip安装"
    echo "----------------------------------------"
    pip uninstall torch torchvision torchaudio -y
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
    python -c "import torch; print(f'PyTorch: {torch.__version__}')"
fi

echo -e "\n=========================================="
echo "修复完成！"
echo "=========================================="
