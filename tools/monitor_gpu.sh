#!/bin/bash
# GPU监控脚本 - 每秒刷新一次

echo "=========================================="
echo "GPU监控（按 Ctrl+C 退出）"
echo "=========================================="
echo ""
echo "观察指标："
echo "  - GPU利用率应该达到 70-90%"
echo "  - 显存使用应该达到 80-90%"
echo "  - 温度应该在合理范围内"
echo ""
echo "=========================================="
echo ""

watch -n 1 'nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | awk -F", " "{printf \"GPU %s (%s)\\n  温度: %s°C | GPU利用率: %s%% | 显存利用率: %s%% | 显存: %sMB/%sMB\\n\\n\", \$1, \$2, \$3, \$4, \$5, \$6, \$7}"'
