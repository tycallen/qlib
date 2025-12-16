"""
将已训练的模型权重添加到现有的 MLflow 实验记录中
"""
import mlflow
from pathlib import Path

# 你的实验和记录 ID
EXPERIMENT_ID = "789877461266984949"
RUN_ID = "f8a1507c4a0d4499b7e70061966557fd"

# 模型文件路径（自动检测是 macOS 还是 Linux）
import os
if os.path.exists("/Volumes/data10"):
    # macOS with mounted volume
    BASE_PATH = "/Volumes/data10/tyc/quant/qlib_own"
else:
    # Linux
    BASE_PATH = "/data10/tyc/quant/qlib_own"

MODEL_PATH = f"{BASE_PATH}/gats_model.pth"
CHECKPOINT_PATH = f"{BASE_PATH}/checkpoints/best_checkpoint.pth"

def add_artifacts_to_run():
    """将模型文件添加到现有的 MLflow run"""

    # 设置 tracking URI（与你的训练脚本一致）
    mlflow.set_tracking_uri("./mlruns")

    # 获取现有的 run
    client = mlflow.tracking.MlflowClient()

    print("="*60)
    print("Adding Model Artifacts to MLflow")
    print("="*60)
    print(f"Experiment ID: {EXPERIMENT_ID}")
    print(f"Run ID: {RUN_ID}")
    print()

    # 检查文件是否存在
    if Path(MODEL_PATH).exists():
        print(f"✓ Found model file: {MODEL_PATH}")
        size_mb = Path(MODEL_PATH).stat().st_size / (1024*1024)
        print(f"  Size: {size_mb:.2f} MB")

        # 添加主模型文件
        client.log_artifact(RUN_ID, MODEL_PATH)
        print(f"✓ Logged to MLflow artifacts")
    else:
        print(f"✗ Model file not found: {MODEL_PATH}")

    print()

    # 检查并添加 checkpoint
    if Path(CHECKPOINT_PATH).exists():
        print(f"✓ Found checkpoint: {CHECKPOINT_PATH}")
        size_mb = Path(CHECKPOINT_PATH).stat().st_size / (1024*1024)
        print(f"  Size: {size_mb:.2f} MB")

        # 添加到 checkpoints 子目录
        client.log_artifact(RUN_ID, CHECKPOINT_PATH, "checkpoints")
        print(f"✓ Logged to MLflow artifacts/checkpoints/")
    else:
        print(f"✗ Checkpoint not found: {CHECKPOINT_PATH}")

    print()
    print("="*60)
    print("Done! Now check your artifacts:")
    print(f"  ls -lh mlruns/{EXPERIMENT_ID}/{RUN_ID}/artifacts/")
    print("="*60)

if __name__ == "__main__":
    add_artifacts_to_run()
