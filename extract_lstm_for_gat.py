"""
提取LSTM权重用于GAT模型初始化

使用方法:
1. 找到训练好的LSTM模型权重文件
2. 运行此脚本提取权重
3. 在GAT配置中使用提取的权重
"""

import torch
import argparse
from pathlib import Path


def extract_lstm_weights(lstm_model_path, output_path=None):
    """
    从完整的LSTM模型中提取基础LSTM层的权重

    Args:
        lstm_model_path: LSTM模型权重文件路径
        output_path: 输出文件路径，如果不指定则使用lstm_model_path
    """
    print(f"加载LSTM模型权重: {lstm_model_path}")

    # 加载模型权重
    checkpoint = torch.load(lstm_model_path, map_location='cpu')

    print("\n原始LSTM模型包含的权重层:")
    for key in checkpoint.keys():
        print(f"  - {key}: {checkpoint[key].shape}")

    # LSTM层的权重包括: rnn.* 和 fc_out.*
    # GAT模型也有这些层，所以可以直接复用

    if output_path is None:
        output_path = lstm_model_path
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存权重（GAT会自动过滤匹配的层）
    torch.save(checkpoint, output_path)
    print(f"\n✓ 权重已保存至: {output_path}")

    return output_path


def inspect_lstm_weights(lstm_model_path):
    """
    检查LSTM模型权重的详细信息

    Args:
        lstm_model_path: LSTM模型权重文件路径
    """
    print(f"检查LSTM模型权重: {lstm_model_path}")

    checkpoint = torch.load(lstm_model_path, map_location='cpu')

    print("\n" + "="*60)
    print("LSTM模型权重详情")
    print("="*60)

    total_params = 0
    for key, value in checkpoint.items():
        params = value.numel()
        total_params += params
        print(f"\n层名称: {key}")
        print(f"  形状: {value.shape}")
        print(f"  参数数量: {params:,}")
        print(f"  数据类型: {value.dtype}")

    print("\n" + "="*60)
    print(f"总参数量: {total_params:,}")
    print("="*60)

    # 检查哪些层可以被GAT使用
    print("\n可用于GAT初始化的层:")
    gat_compatible_layers = []
    for key in checkpoint.keys():
        if key.startswith('rnn.') or key.startswith('fc_out.'):
            gat_compatible_layers.append(key)
            print(f"  ✓ {key}")

    if not gat_compatible_layers:
        print("  警告: 没有找到可用于GAT的权重层!")

    return checkpoint


def create_gat_config_example(lstm_model_path, output_yaml="workflow_config_gat_from_lstm.yaml"):
    """
    创建使用LSTM权重的GAT配置文件示例

    Args:
        lstm_model_path: LSTM模型权重路径
        output_yaml: 输出配置文件路径
    """
    config_content = f"""qlib_init:
    provider_uri: "~/.qlib/qlib_data/cn_data"
    region: cn
market: csi300
benchmark: SH000300
data_handler_config: &data_handler_config
    start_time: 2008-01-01
    end_time: 2020-08-01
    fit_start_time: 2008-01-01
    fit_end_time: 2014-12-31
    instruments: *market
port_analysis_config: &port_analysis_config
    strategy:
        class: TopkDropoutStrategy
        module_path: qlib.contrib.strategy
        kwargs:
            signal: <PRED>
            topk: 50
            n_drop: 5
    backtest:
        start_time: 2017-01-01
        end_time: 2020-08-01
        account: 100000000
        benchmark: *benchmark
        exchange_kwargs:
            limit_threshold: 0.095
            deal_price: close
            open_cost: 0.0005
            close_cost: 0.0015
            min_cost: 5
task:
    model:
        class: GATs
        module_path: qlib.contrib.model.pytorch_gats
        kwargs:
            d_feat: 6
            hidden_size: 64
            num_layers: 2
            dropout: 0.0
            n_epochs: 200
            lr: 0.001
            early_stop: 20
            batch_size: 800
            metric: loss
            loss: mse
            base_model: "LSTM"  # 使用LSTM作为base模型
            model_path: "{lstm_model_path}"  # LSTM预训练权重路径
            optimizer: adam
            GPU: 0
    dataset:
        class: DatasetH
        module_path: qlib.data.dataset
        kwargs:
            handler:
                class: Alpha360
                module_path: qlib.contrib.data.handler
                kwargs: *data_handler_config
            segments:
                train: [2008-01-01, 2014-12-31]
                valid: [2015-01-01, 2016-12-31]
                test: [2017-01-01, 2020-08-01]
    record:
        - class: SignalRecord
          module_path: qlib.workflow.record_temp
          kwargs: {{}}
        - class: SigAnaRecord
          module_path: qlib.workflow.record_temp
          kwargs:
              ana_long_short: False
              ann_scaler: 252
        - class: PortAnaRecord
          module_path: qlib.workflow.record_temp
          kwargs:
              config: *port_analysis_config
"""

    with open(output_yaml, 'w', encoding='utf-8') as f:
        f.write(config_content)

    print(f"\n✓ GAT配置文件已创建: {output_yaml}")
    print(f"  配置使用LSTM权重路径: {lstm_model_path}")
    print(f"\n使用方式:")
    print(f"  qlib_workflow --config_path {output_yaml}")


def main():
    parser = argparse.ArgumentParser(description='提取LSTM权重用于GAT模型')
    parser.add_argument('--lstm_path', type=str, required=True,
                      help='LSTM模型权重文件路径')
    parser.add_argument('--output', type=str, default=None,
                      help='输出权重文件路径（可选）')
    parser.add_argument('--inspect', action='store_true',
                      help='检查LSTM权重详细信息')
    parser.add_argument('--create_config', action='store_true',
                      help='创建GAT配置文件示例')

    args = parser.parse_args()

    # 检查文件是否存在
    if not Path(args.lstm_path).exists():
        print(f"错误: LSTM模型文件不存在: {args.lstm_path}")
        return

    # 检查权重
    if args.inspect:
        inspect_lstm_weights(args.lstm_path)

    # 提取权重
    output_path = extract_lstm_weights(args.lstm_path, args.output)

    # 创建配置文件
    if args.create_config:
        create_gat_config_example(str(output_path))

    print("\n" + "="*60)
    print("使用说明:")
    print("="*60)
    print("\n方式1: 使用配置文件")
    print("  修改上面生成的YAML配置文件，然后运行:")
    print("  qlib_workflow --config_path workflow_config_gat_from_lstm.yaml")

    print("\n方式2: 使用Python代码")
    print(f"""
from qlib.contrib.model.pytorch_gats import GATs

# 创建GAT模型，指定使用LSTM作为base模型
model = GATs(
    d_feat=6,
    hidden_size=64,
    num_layers=2,
    dropout=0.0,
    base_model="LSTM",           # 使用LSTM作为base模型
    model_path="{output_path}",  # LSTM预训练权重路径
    lr=0.001,
    n_epochs=200,
)

# 训练模型（会自动加载LSTM权重初始化）
model.fit(dataset, save_path="gat_model.pkl")
""")


if __name__ == "__main__":
    main()
