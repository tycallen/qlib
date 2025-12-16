# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Enhanced GATs with:
1. Best checkpoint saving
2. Ctrl+C graceful handling with best checkpoint recovery
3. Enhanced training logging
4. MLflow integration
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import copy
import signal
import sys
import time
from pathlib import Path
from ...utils import get_or_create_path
from ...log import get_module_logger
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Sampler

from .pytorch_utils import count_parameters
from ...model.base import Model
from ...data.dataset.handler import DataHandlerLP
from .pytorch_lstm import LSTMModel
from .pytorch_gru import GRUModel

# Try to import MLflow
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class DailyBatchSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        # calculate number of samples in each batch
        self.daily_count = (
            pd.Series(index=self.data_source.get_index()).groupby("datetime", group_keys=False).size().values
        )
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)  # calculate begin index of each batch
        self.daily_index[0] = 0

    def __iter__(self):
        for idx, count in zip(self.daily_index, self.daily_count):
            yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source)


class GATs(Model):
    """Enhanced GATs Model with checkpoint management and MLflow logging

    New Features:
    - Automatic best checkpoint saving
    - Graceful Ctrl+C handling with checkpoint recovery
    - Enhanced training logs with progress bar
    - MLflow integration for experiment tracking
    - Periodic checkpoint saving

    Parameters
    ----------
    lr : float
        learning rate
    d_feat : int
        input dimensions for each time step
    metric : str
        the evaluation metric used in early stop
    optimizer : str
        optimizer name
    GPU : int
        the GPU ID used for training
    checkpoint_dir : str
        directory to save checkpoints (default: "./checkpoints")
    save_checkpoint_interval : int
        save checkpoint every N epochs (default: 5)
    use_mlflow : bool
        whether to use MLflow for logging (default: True if available)
    """

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
        checkpoint_dir="./checkpoints",
        save_checkpoint_interval=5,
        use_mlflow=True,
        **kwargs,
    ):
        # Set logger.
        self.logger = get_module_logger("GATs_Enhanced")
        self.logger.info("GATs Enhanced pytorch version...")

        # set hyper-parameters.
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
        if self.use_mlflow:
            self.logger.info("MLflow logging enabled")
        elif use_mlflow and not MLFLOW_AVAILABLE:
            self.logger.warning("MLflow requested but not available. Install with: pip install mlflow")

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

        # Training state (for interruption handling)
        self.interrupted = False
        self.best_state = None
        self.best_score = -np.inf
        self.best_epoch = 0

        self.logger.info(
            "="*60 + "\n"
            "GATs Enhanced Parameters:\n"
            "="*60 + "\n"
            f"  Model Architecture:\n"
            f"    - d_feat: {d_feat}\n"
            f"    - hidden_size: {hidden_size}\n"
            f"    - num_layers: {num_layers}\n"
            f"    - dropout: {dropout}\n"
            f"    - base_model: {base_model}\n"
            f"  Training:\n"
            f"    - n_epochs: {n_epochs}\n"
            f"    - lr: {lr}\n"
            f"    - optimizer: {optimizer.lower()}\n"
            f"    - loss: {loss}\n"
            f"    - metric: {metric}\n"
            f"    - early_stop: {early_stop}\n"
            f"  Hardware:\n"
            f"    - device: {self.device}\n"
            f"    - GPUs: {self.gpus}\n"
            f"    - use_AMP: {self.use_amp}\n"
            f"  Checkpoint:\n"
            f"    - checkpoint_dir: {self.checkpoint_dir}\n"
            f"    - save_interval: {self.save_checkpoint_interval} epochs\n"
            f"  Logging:\n"
            f"    - use_mlflow: {self.use_mlflow}\n"
            f"    - seed: {seed}\n"
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

        self.logger.info("model:\n{:}".format(self.GAT_model))
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
            self.logger.info("AMP (Automatic Mixed Precision) training enabled")

        self.fitted = False

        # Setup signal handler for Ctrl+C
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
        self.logger.info(f"  ✓ Checkpoint saved: {filename}")
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
        self.GAT_model.train()
        epoch_loss = 0
        batch_count = 0

        for data in data_loader:
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    data = data.squeeze()
                    feature = data[:, :, 0:-1].to(self.device)
                    label = data[:, -1, -1].to(self.device)
                    pred = self.GAT_model(feature.float())
                    loss = self.loss_fn(pred, label)

                self.train_optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.train_optimizer)
                torch.nn.utils.clip_grad_value_(self.GAT_model.parameters(), 3.0)
                self.scaler.step(self.train_optimizer)
                self.scaler.update()
            else:
                data = data.squeeze()
                feature = data[:, :, 0:-1].to(self.device)
                label = data[:, -1, -1].to(self.device)
                pred = self.GAT_model(feature.float())
                loss = self.loss_fn(pred, label)

                self.train_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.GAT_model.parameters(), 3.0)
                self.train_optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        return epoch_loss / batch_count if batch_count > 0 else 0

    def test_epoch(self, data_loader):
        self.GAT_model.eval()
        scores = []
        losses = []

        for data in data_loader:
            data = data.squeeze()
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

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
        """Enhanced fit with checkpoint management and MLflow logging"""

        # Log to MLflow if enabled
        if self.use_mlflow:
            mlflow.log_params({
                "d_feat": self.d_feat,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "lr": self.lr,
                "optimizer": self.optimizer,
                "base_model": self.base_model,
                "n_epochs": self.n_epochs,
                "early_stop": self.early_stop,
            })

        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        dl_train.config(fillna_type="ffill+bfill")
        dl_valid.config(fillna_type="ffill+bfill")

        sampler_train = DailyBatchSampler(dl_train)
        sampler_valid = DailyBatchSampler(dl_valid)

        train_loader = DataLoader(dl_train, sampler=sampler_train, num_workers=self.n_jobs, drop_last=True)
        valid_loader = DataLoader(dl_valid, sampler=sampler_valid, num_workers=self.n_jobs, drop_last=True)

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
                    f"Epoch {step+1}/{self.n_epochs} | Time: {epoch_time:.2f}s\n"
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
                    }, step=step)

                evals_result["train"].append(train_score)
                evals_result["valid"].append(val_score)

                # Update best model
                if val_score > self.best_score:
                    self.best_score = val_score
                    stop_steps = 0
                    self.best_epoch = step
                    self.best_state = copy.deepcopy(self.GAT_model.state_dict())

                    # Save best checkpoint
                    self.save_checkpoint(step, val_score, is_best=True)
                    self.logger.info("  ★ New best model!")
                else:
                    stop_steps += 1

                # Periodic checkpoint saving
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
            self.logger.warning("Training interrupted by user")
            self.logger.warning("="*60)

        finally:
            # Always restore best model
            training_time = time.time() - training_start_time

            self.logger.info("\n" + "="*60)
            self.logger.info("Training Summary")
            self.logger.info("="*60)
            self.logger.info(f"  Total Time: {training_time/60:.2f} minutes")
            self.logger.info(f"  Best Score: {self.best_score:.6f} @ Epoch {self.best_epoch+1}")
            self.logger.info(f"  Status: {'Interrupted' if self.interrupted else 'Completed'}")
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
                mlflow.log_artifact(str(save_path))

            if self.use_gpu:
                torch.cuda.empty_cache()

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        sampler_test = DailyBatchSampler(dl_test)
        test_loader = DataLoader(dl_test, sampler=sampler_test, num_workers=self.n_jobs)
        self.GAT_model.eval()
        preds = []

        for data in test_loader:
            data = data.squeeze()
            feature = data[:, :, 0:-1].to(self.device)

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
        return self.fc_out(hidden).squeeze()
