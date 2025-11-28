"""
快速加载模型并进行推理的简化脚本

使用方法:
    python quick_predict.py --model_path <模型路径> --start_date 2020-01-01 --end_date 2020-12-31
"""

import argparse
import pickle
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config


def quick_predict(model_path, start_date, end_date, output_file="predictions.csv"):
    """
    快速预测函数

    Args:
        model_path: 模型pkl文件路径
        start_date: 开始日期
        end_date: 结束日期
        output_file: 输出文件路径
    """
    print("="*60)
    print("快速预测")
    print("="*60)

    # 初始化Qlib
    print("\n初始化Qlib...")
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

    # 加载模型
    print(f"\n加载模型: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"✓ 模型加载成功: {type(model).__name__}")

    # 创建数据集配置
    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha360",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": "2008-01-01",
                    "end_time": end_date,
                    "fit_start_time": "2008-01-01",
                    "fit_end_time": "2014-12-31",
                    "instruments": "csi300",
                },
            },
            "segments": {
                "train": ("2008-01-01", "2014-12-31"),
                "valid": ("2015-01-01", "2016-12-31"),
                "test": (start_date, end_date),
            },
        },
    }

    # 创建数据集
    print(f"\n准备数据: {start_date} 至 {end_date}")
    dataset = init_instance_by_config(dataset_config)

    # 预测
    print("\n正在预测...")
    pred = model.predict(dataset, segment="test")

    # 显示统计信息
    print(f"\n✓ 预测完成")
    print(f"  样本数: {len(pred)}")
    print(f"  均值: {pred.mean():.6f}")
    print(f"  标准差: {pred.std():.6f}")
    print(f"  范围: [{pred.min():.6f}, {pred.max():.6f}]")

    # 保存结果
    print(f"\n保存到: {output_file}")
    pred_df = pred.reset_index()
    pred_df.columns = ['datetime', 'instrument', 'score']
    pred_df.to_csv(output_file, index=False)

    print(f"\n✓ 完成！")

    return pred


def main():
    parser = argparse.ArgumentParser(description='快速加载模型并预测')
    parser.add_argument('--model_path', type=str,
                      default='./mlruns/686249546704553370/2ee5b9c2d8f2420e8cb39aa84b35546e/artifacts/params.pkl',
                      help='模型pkl文件路径')
    parser.add_argument('--start_date', type=str, default='2020-01-01',
                      help='预测开始日期')
    parser.add_argument('--end_date', type=str, default='2020-12-31',
                      help='预测结束日期')
    parser.add_argument('--output', type=str, default='predictions.csv',
                      help='输出文件路径')

    args = parser.parse_args()

    quick_predict(args.model_path, args.start_date, args.end_date, args.output)


if __name__ == "__main__":
    main()
