#!/bin/bash
# 实时监控训练进度

clear
echo "=========================================="
echo "GATs 训练实时监控"
echo "=========================================="
echo "按 Ctrl+C 退出监控（不会停止训练）"
echo ""

while true; do
    clear
    echo "=========================================="
    echo "GATs 训练实时监控 - $(date +"%Y-%m-%d %H:%M:%S")"
    echo "=========================================="
    echo ""

    # GPU状态
    echo "🎮 GPU 状态："
    nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | \
      awk -F", " '{
        printf "  GPU %s: ", $1;
        if ($2 < 30) printf "❌ ";
        else if ($2 < 70) printf "⚠️  ";
        else printf "✅ ";
        printf "利用率=%s%% | 显存=%sMB/%sMB (%.0f%%) | 温度=%s°C | 功耗=%sW\n",
               $2, $4, $5, ($4/$5)*100, $6, int($7);
      }'
    echo ""

    # 训练进度
    echo "📊 训练进度："
    if grep -q "Epoch" training.log 2>/dev/null; then
        tail -15 training.log | grep -E "Epoch|train|valid|score|Best|Stop" | tail -10 | sed 's/^/  /'
    else
        echo "  等待第一个Epoch完成..."
    fi
    echo ""

    # 进程状态
    echo "💻 进程状态："
    ps aux | grep "train_gats_enhanced_example.py" | grep -v grep | \
      awk '{printf "  CPU: %s%% | 内存: %sMB | 运行时间: %s\n", $3, int($6/1024), $10}' || \
      echo "  ⚠️  训练进程未运行"
    echo ""

    echo "=========================================="
    echo "刷新间隔: 3秒 | 按 Ctrl+C 退出"
    echo "=========================================="

    sleep 3
done
