#!/bin/bash
# 训练状态检查脚本

echo "=========================================="
echo "GATs 训练状态检查"
echo "=========================================="
echo ""

# 1. 检查进程
echo "📌 训练进程状态："
ps aux | grep "train_gats_enhanced_example.py" | grep -v grep | awk '{printf "  PID: %s | CPU: %s%% | 内存: %sMB | 运行时间: %s\n", $2, $3, int($6/1024), $10}'
echo ""

# 2. 检查GPU
echo "🎮 GPU使用情况："
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
  awk -F", " '{printf "  GPU %s: 利用率=%s%% | 显存=%sMB/%sMB\n", $1, $2, $3, $4}'
echo ""

# 3. 检查日志
echo "📝 最新日志 (最后10行)："
tail -10 training.log | sed 's/^/  /'
echo ""

echo "=========================================="
echo "💡 提示："
echo "  - 如果GPU利用率为0，表示还在加载数据"
echo "  - 如果看到 'Epoch' 字样，表示训练已开始"
echo "  - 使用 'tail -f training.log' 实时查看日志"
echo "=========================================="
