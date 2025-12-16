# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Optimized GAT_ts for High GPU Utilization

Key optimizations:
1. Configurable batch strategy (daily vs fixed-size)
2. Pin memory for faster data transfer
3. Persistent workers to reduce overhead
4. Prefetch factor for data loading
5. Mixed precision training support
6. Gradient accumulation for effective larger batches
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import copy
import signal
import time
from pathlib import Path
from ...utils import get_or_create_path
from ...log import get_module_logger
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler

from .pytorch_utils import count_parameters
from ...model.base import Model
from ...data.dataset.handler import DataHandlerLP
from .pytorch_lstm import LSTMModel
from .pytorch_gru import GRUModel

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class DailyBatchSampler(Sampler):
    """Daily batch sampler - each batch contains one day's data"""
    def __init__(self, data_source):
        self.data_source = data_source
        self.daily_count = (
            pd.Series(index=self.data_source.get_index())
            .groupby("datetime", group_keys=False).size().values
        )
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0

    def __iter__(self):
        for idx, count in zip(self.daily_index, self.daily_count):
            yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.daily_index)


class FixedSizeBatchSampler(Sampler):
    """Fixed size batch sampler for better GPU utilization"""
    def __init__(self, data_source, batch_size, drop_last=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_samples = len(data_source)

    def __iter__(self):
        indices = list(range(self.num_samples))
        # Shuffle indices for better training
        np.random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size


class GATs(Model):
    """Optimized GAT_ts Model for High GPU Utilization"""

    def __init__(
        self,
        d_feat=20,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        lr=0.001,
        metric="",
        early_stop=20,
        loss="mse",
        base_model="GRU",
        model_path=None,
        optimizer="adam",
        GPU=0,
        gpus=[0],
        use_amp=False,
        n_jobs=10,
        seed=None,
        # Checkpoint settings
        checkpoint_dir="./checkpoints",
        save_checkpoint_interval=5,
        use_mlflow=True,
        # GPU optimization settings
        batch_strategy="daily",  # "daily" or "fixed"
        fixed_batch_size=2000,   # used when batch_strategy="fixed"
        pin_memory=True,         # faster data transfer
        persistent_workers=True, # reduce worker respawn overhead
        prefetch_factor=2,       # prefetch batches
        gradient_accumulation_steps=1,  # simulate larger batch
        **kwargs,
    ):
        # Set logger
        self.logger = get_module_logger("GATs_Optimized")
        self.logger.info("GATs Optimized for High GPU Utilization...")

        # set hyper-parameters
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.base_model = base_model
        self.model_path = model_path

        # Checkpoint settings
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_checkpoint_interval = save_checkpoint_interval

        # MLflow settings
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE

        # GPU optimization settings
        self.batch_strategy = batch_strategy
        self.fixed_batch_size = fixed_batch_size
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Handle GPU configuration
        if isinstance(gpus, list):
            self.gpus = gpus
        else:
            self.gpus = [GPU] if GPU >= 0 else []

        # Set device
        if torch.cuda.is_available() and len(self.gpus) > 0:
            self.device = torch.device(f"cuda:{self.gpus[0]}")
        else:
            self.device = torch.device("cpu")

        self.n_jobs = n_jobs
        self.seed = seed
        self.use_amp = use_amp

        # Training state
        self.interrupted = False
        self.best_state = None
        self.best_score = -np.inf
        self.best_epoch = 0

        self.logger.info(
            "="*60 + "\n"
            "GATs Optimized Configuration:\n"
            "="*60 + "\n"
            f"  Model:\n"
            f"    - d_feat: {d_feat}\n"
            f"    - hidden_size: {hidden_size}\n"
            f"    - num_layers: {num_layers}\n"
            f"    - base_model: {base_model}\n"
            f"  Training:\n"
            f"    - n_epochs: {n_epochs}\n"
            f"    - lr: {lr}\n"
            f"    - optimizer: {optimizer}\n"
            f"  GPU Optimization:\n"
            f"    - batch_strategy: {batch_strategy}\n"
            f"    - fixed_batch_size: {fixed_batch_size}\n"
            f"    - gradient_accumulation: {gradient_accumulation_steps}\n"
            f"    - pin_memory: {pin_memory}\n"
            f"    - persistent_workers: {persistent_workers}\n"
            f"    - prefetch_factor: {prefetch_factor}\n"
            f"    - num_workers: {n_jobs}\n"
            f"    - use_AMP: {use_amp}\n"
            f"  Hardware:\n"
            f"    - device: {self.device}\n"
            f"    - GPUs: {self.gpus}\n"
            "="*60
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.GAT_model = GATModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            base_model=self.base_model,
        )

        # Move model to device
        self.GAT_model.to(self.device)

        # Wrap with DataParallel if multiple GPUs
        if len(self.gpus) > 1:
            self.GAT_model = nn.DataParallel(self.GAT_model, device_ids=self.gpus)
            self.logger.info(f"Using DataParallel with {len(self.gpus)} GPUs")

        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.GAT_model)))

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.GAT_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.GAT_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        # Initialize GradScaler for AMP if enabled
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            self.logger.info("AMP enabled")

        self.fitted = False

        # Setup signal handler
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        self.logger.warning("\n" + "="*60)
        self.logger.warning("Ctrl+C detected! Saving best checkpoint...")
        self.logger.warning("="*60)
        self.interrupted = True

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred - label) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])
        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)
        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])
        raise ValueError("unknown metric `%s`" % self.metric)

    def save_checkpoint(self, epoch, score, is_best=False, filename=None):
        """Save model checkpoint"""
        if filename is None:
            if is_best:
                filename = self.checkpoint_dir / "best_checkpoint.pth"
            else:
                filename = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.GAT_model.state_dict(),
            'optimizer_state_dict': self.train_optimizer.state_dict(),
            'score': score,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
        }

        torch.save(checkpoint, filename)
        return filename

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.GAT_model.load_state_dict(checkpoint['model_state_dict'])
        self.train_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_score = checkpoint['best_score']
        self.best_epoch = checkpoint['best_epoch']

        self.logger.info(f"  ✓ Checkpoint loaded (epoch {checkpoint['epoch']}, score {checkpoint['score']:.6f})")
        return checkpoint

    def train_epoch(self, data_loader):
        """Train one epoch with gradient accumulation support"""
        self.GAT_model.train()
        epoch_loss = 0
        batch_count = 0

        self.train_optimizer.zero_grad()

        for batch_idx, data in enumerate(data_loader):
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    data = data.squeeze()
                    feature = data[:, :, 0:-1].to(self.device, non_blocking=True)
                    label = data[:, -1, -1].to(self.device, non_blocking=True)
                    pred = self.GAT_model(feature.float())
                    loss = self.loss_fn(pred, label)
                    loss = loss / self.gradient_accumulation_steps  # scale loss

                self.scaler.scale(loss).backward()

                # Update weights every gradient_accumulation_steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.train_optimizer)
                    torch.nn.utils.clip_grad_value_(self.GAT_model.parameters(), 3.0)
                    self.scaler.step(self.train_optimizer)
                    self.scaler.update()
                    self.train_optimizer.zero_grad()
            else:
                data = data.squeeze()
                feature = data[:, :, 0:-1].to(self.device, non_blocking=True)
                label = data[:, -1, -1].to(self.device, non_blocking=True)
                pred = self.GAT_model(feature.float())
                loss = self.loss_fn(pred, label)
                loss = loss / self.gradient_accumulation_steps  # scale loss

                loss.backward()

                # Update weights every gradient_accumulation_steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_value_(self.GAT_model.parameters(), 3.0)
                    self.train_optimizer.step()
                    self.train_optimizer.zero_grad()

            epoch_loss += loss.item() * self.gradient_accumulation_steps
            batch_count += 1

        return epoch_loss / batch_count if batch_count > 0 else 0

    def test_epoch(self, data_loader):
        self.GAT_model.eval()
        scores = []
        losses = []

        for data in data_loader:
            data = data.squeeze()
            feature = data[:, :, 0:-1].to(self.device, non_blocking=True)
            label = data[:, -1, -1].to(self.device, non_blocking=True)

            with torch.no_grad():
                pred = self.GAT_model(feature.float())
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        dataset,
        evals_result=dict(),
        save_path=None,
    ):
        """Enhanced fit with GPU optimization"""

        # Log to MLflow
        if self.use_mlflow:
            mlflow.log_params({
                "d_feat": self.d_feat,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "batch_strategy": self.batch_strategy,
                "fixed_batch_size": self.fixed_batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "n_jobs": self.n_jobs,
                "pin_memory": self.pin_memory,
            })

        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        dl_train.config(fillna_type="ffill+bfill")
        dl_valid.config(fillna_type="ffill+bfill")

        # Choose batch strategy
        if self.batch_strategy == "daily":
            self.logger.info("Using daily batch strategy")
            sampler_train = DailyBatchSampler(dl_train)
            sampler_valid = DailyBatchSampler(dl_valid)
        else:  # fixed
            self.logger.info(f"Using fixed batch strategy (batch_size={self.fixed_batch_size})")
            sampler_train = FixedSizeBatchSampler(dl_train, self.fixed_batch_size, drop_last=True)
            sampler_valid = FixedSizeBatchSampler(dl_valid, self.fixed_batch_size, drop_last=True)

        # Optimized DataLoader settings
        dataloader_kwargs = {
            "num_workers": self.n_jobs,
            "pin_memory": self.pin_memory and self.use_gpu,
            "drop_last": True,
        }

        # Add persistent_workers and prefetch_factor if PyTorch >= 1.7
        import torch
        pytorch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
        if pytorch_version >= (1, 7):
            dataloader_kwargs["persistent_workers"] = self.persistent_workers and self.n_jobs > 0
            dataloader_kwargs["prefetch_factor"] = self.prefetch_factor

        train_loader = DataLoader(dl_train, sampler=sampler_train, **dataloader_kwargs)
        valid_loader = DataLoader(dl_valid, sampler=sampler_valid, **dataloader_kwargs)

        self.logger.info(f"DataLoader settings: {dataloader_kwargs}")

        save_path = get_or_create_path(save_path)

        stop_steps = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # Load pretrained base_model
        if self.base_model == "LSTM":
            pretrained_model = LSTMModel(d_feat=self.d_feat, hidden_size=self.hidden_size, num_layers=self.num_layers)
        elif self.base_model == "GRU":
            pretrained_model = GRUModel(d_feat=self.d_feat, hidden_size=self.hidden_size, num_layers=self.num_layers)
        else:
            raise ValueError("unknown base model name `%s`" % self.base_model)

        if self.model_path is not None:
            self.logger.info("Loading pretrained model...")
            pretrained_model.load_state_dict(torch.load(self.model_path, map_location=self.device))

        model_dict = self.GAT_model.state_dict()
        pretrained_dict = {
            k: v for k, v in pretrained_model.state_dict().items() if k in model_dict
        }
        model_dict.update(pretrained_dict)
        self.GAT_model.load_state_dict(model_dict)
        self.logger.info("Loading pretrained model Done...")

        # Train
        self.logger.info("\n" + "="*60)
        self.logger.info("Starting Training")
        self.logger.info("="*60)
        self.fitted = True

        training_start_time = time.time()

        try:
            for step in range(self.n_epochs):
                if self.interrupted:
                    break

                epoch_start_time = time.time()

                # Train
                train_loss = self.train_epoch(train_loader)

                # Evaluate
                train_loss_eval, train_score = self.test_epoch(train_loader)
                val_loss, val_score = self.test_epoch(valid_loader)

                epoch_time = time.time() - epoch_start_time

                # Enhanced logging
                self.logger.info(
                    f"\n{'='*60}\n"
                    f"Epoch {step+1}/{self.n_epochs} | Time: {epoch_time:.2f}s | {1/epoch_time*len(train_loader):.1f} batch/s\n"
                    f"{'='*60}\n"
                    f"  Train Loss: {train_loss:.6f} | Train Score: {train_score:.6f}\n"
                    f"  Valid Loss: {val_loss:.6f} | Valid Score: {val_score:.6f}\n"
                    f"  Best Score: {self.best_score:.6f} @ Epoch {self.best_epoch+1}\n"
                    f"  Stop Steps: {stop_steps}/{self.early_stop}"
                )

                # Log to MLflow
                if self.use_mlflow:
                    mlflow.log_metrics({
                        "train_loss": train_loss,
                        "train_score": train_score,
                        "val_loss": val_loss,
                        "val_score": val_score,
                        "epoch_time": epoch_time,
                        "throughput_batch_per_sec": 1/epoch_time*len(train_loader),
                    }, step=step)

                evals_result["train"].append(train_score)
                evals_result["valid"].append(val_score)

                # Update best model
                if val_score > self.best_score:
                    self.best_score = val_score
                    stop_steps = 0
                    self.best_epoch = step
                    self.best_state = copy.deepcopy(self.GAT_model.state_dict())
                    self.save_checkpoint(step, val_score, is_best=True)
                    self.logger.info("  ★ New best model!")
                else:
                    stop_steps += 1

                # Periodic checkpoint
                if (step + 1) % self.save_checkpoint_interval == 0:
                    self.save_checkpoint(step, val_score, is_best=False)

                # Early stopping
                if stop_steps >= self.early_stop:
                    self.logger.info("\n" + "="*60)
                    self.logger.info("Early stopping triggered")
                    self.logger.info("="*60)
                    break

        except KeyboardInterrupt:
            self.interrupted = True
            self.logger.warning("\n" + "="*60)
            self.logger.warning("Training interrupted")
            self.logger.warning("="*60)

        finally:
            training_time = time.time() - training_start_time

            self.logger.info("\n" + "="*60)
            self.logger.info("Training Summary")
            self.logger.info("="*60)
            self.logger.info(f"  Total Time: {training_time/60:.2f} minutes")
            self.logger.info(f"  Best Score: {self.best_score:.6f} @ Epoch {self.best_epoch+1}")
            self.logger.info("="*60)

            if self.best_state is not None:
                self.logger.info("Restoring best model...")
                self.GAT_model.load_state_dict(self.best_state)
                torch.save(self.best_state, save_path)
                self.logger.info(f"  ✓ Best model saved to: {save_path}")

            if self.use_mlflow:
                mlflow.log_metrics({
                    "best_score": self.best_score,
                    "best_epoch": self.best_epoch,
                    "total_training_time": training_time,
                })

            if self.use_gpu:
                torch.cuda.empty_cache()

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")

        if self.batch_strategy == "daily":
            sampler_test = DailyBatchSampler(dl_test)
        else:
            sampler_test = FixedSizeBatchSampler(dl_test, self.fixed_batch_size, drop_last=False)

        test_loader = DataLoader(
            dl_test,
            sampler=sampler_test,
            num_workers=self.n_jobs,
            pin_memory=self.pin_memory and self.use_gpu
        )

        self.GAT_model.eval()
        preds = []

        for data in test_loader:
            data = data.squeeze()
            feature = data[:, :, 0:-1].to(self.device, non_blocking=True)

            with torch.no_grad():
                pred = self.GAT_model(feature.float()).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=dl_test.get_index())


class GATModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU"):
        super().__init__()

        if base_model == "GRU":
            self.rnn = nn.GRU(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        elif base_model == "LSTM":
            self.rnn = nn.LSTM(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            raise ValueError("unknown base model name `%s`" % base_model)

        self.hidden_size = hidden_size
        self.d_feat = d_feat
        self.transformation = nn.Linear(self.hidden_size, self.hidden_size)
        self.a = nn.Parameter(torch.randn(self.hidden_size * 2, 1))
        self.a.requires_grad = True
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def cal_attention(self, x, y):
        x = self.transformation(x)
        y = self.transformation(y)

        sample_num = x.shape[0]
        dim = x.shape[1]
        e_x = x.expand(sample_num, sample_num, dim)
        e_y = torch.transpose(e_x, 0, 1)
        attention_in = torch.cat((e_x, e_y), 2).view(-1, dim * 2)
        self.a_t = torch.t(self.a)
        attention_out = self.a_t.mm(torch.t(attention_in)).view(sample_num, sample_num)
        attention_out = self.leaky_relu(attention_out)
        att_weight = self.softmax(attention_out)
        return att_weight

    def forward(self, x):
        out, _ = self.rnn(x)
        hidden = out[:, -1, :]
        att_weight = self.cal_attention(hidden, hidden)
        hidden = att_weight.mm(hidden) + hidden
        hidden = self.fc(hidden)
        hidden = self.leaky_relu(hidden)
        # 使用squeeze(-1)而不是squeeze()，避免在DataParallel下出现维度问题
        return self.fc_out(hidden).squeeze(-1)
