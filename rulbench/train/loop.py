"""Unified training loop.

There is exactly one trainer. The proposed model, every deep baseline and every
ablation run through it with the same optimiser, schedule, loss and stopping
rule; only the config changes. This removes the confound in the thesis version,
where the proposed model was trained full-batch with a weighted loss while all
comparison models were trained mini-batch with a plain MSE.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from ..config import Config
from ..losses import build_supervised_loss, multi_kernel_mmd
from ..utils import get_logger, seed_everything

log = get_logger()


@dataclass
class TrainResult:
    """Everything a downstream report needs about one training run."""

    best_val_loss: float
    best_epoch: int
    epochs_run: int
    state_dict: Dict[str, torch.Tensor]
    history: Dict[str, list] = field(default_factory=dict)


def _as_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float32, device=device)


def _cycle(loader: DataLoader):
    """Yield batches from ``loader`` forever, reshuffling between epochs."""
    while True:
        for batch in loader:
            yield batch


class Trainer:
    """Config-driven trainer with early stopping and checkpoint restore."""

    def __init__(self, model: nn.Module, cfg: Config, device: torch.device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device

        tcfg = cfg.get_path("train", {}) or {}
        self.epochs = int(tcfg.get("epochs", 150))
        self.batch_size = int(tcfg.get("batch_size", 128))
        self.grad_clip = tcfg.get("grad_clip", None)
        self.patience = int(tcfg.get("patience", 15))
        self.use_weighted_loss = tcfg.get("loss", "weighted_mse")

        loss_params = tcfg.get("loss_params", {}) or {}
        self.loss_params = {
            "threshold": float(loss_params.get("threshold", 0.3)),
            "weight": float(loss_params.get("weight", 2.0)),
        }
        self.supervised = build_supervised_loss(self.use_weighted_loss, **self.loss_params)

        # model selection criterion; defaults to plain MSE on the validation
        # split, matching the thesis, but can be aligned with the training loss
        self.val_loss_name = tcfg.get("val_loss", "mse")
        self.val_loss = (
            nn.functional.mse_loss
            if self.val_loss_name == "mse"
            else build_supervised_loss(self.val_loss_name, **self.loss_params)
        )

        opt_cfg = tcfg.get("optim", {}) or {}
        self.lr = float(opt_cfg.get("lr", 1e-3))
        self.weight_decay = float(opt_cfg.get("weight_decay", 0.0))
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        sch_cfg = tcfg.get("scheduler", {}) or {}
        self.scheduler = None
        if sch_cfg.get("name", "reduce_on_plateau") == "reduce_on_plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=float(sch_cfg.get("factor", 0.5)),
                patience=int(sch_cfg.get("patience", 5)),
                min_lr=float(sch_cfg.get("min_lr", 1e-6)),
            )

        mmd_cfg = tcfg.get("mmd", {}) or {}
        self.lambda_mmd = float(mmd_cfg.get("lambda", 0.1))
        self.lambda_schedule = mmd_cfg.get("schedule", None)
        self.mmd_kwargs = {
            "kernel_mul": float(mmd_cfg.get("kernel_mul", 2.0)),
            "kernel_num": int(mmd_cfg.get("kernel_num", 5)),
        }

    # ------------------------------------------------------------------ data
    def _loaders(
        self,
        train: Dict[str, np.ndarray],
        val: Dict[str, np.ndarray],
        target: Optional[Dict[str, np.ndarray]],
    ):
        batch = min(self.batch_size, int(train["X"].shape[0]))
        ds = TensorDataset(
            _as_tensor(train["X"], self.device),
            _as_tensor(train["ratio"], self.device).unsqueeze(-1),
        )
        train_loader = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)
        val_ds = TensorDataset(
            _as_tensor(val["X"], self.device),
            _as_tensor(val["ratio"], self.device).unsqueeze(-1),
        )
        val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False)

        target_loader = None
        if target is not None:
            tgt_ds = TensorDataset(_as_tensor(target["X"], self.device))
            # draw exactly one source-sized batch from the target pool, with
            # replacement, so the two loaders can be paired batch by batch
            # regardless of how the split sizes compare
            target_loader = DataLoader(
                tgt_ds,
                batch_size=batch,
                sampler=RandomSampler(tgt_ds, replacement=True, num_samples=batch),
                drop_last=False,
            )
        return train_loader, val_loader, target_loader

    # ------------------------------------------------------------------- fit
    def fit(
        self,
        train: Dict[str, np.ndarray],
        val: Dict[str, np.ndarray],
        target: Optional[Dict[str, np.ndarray]] = None,
        seed: int = 42,
        tag: str = "",
    ) -> TrainResult:
        seed_everything(seed)
        train_loader, val_loader, target_loader = self._loaders(train, val, target)

        supports_mmd = hasattr(self.model, "mmd_representation") and target_loader is not None
        target_stream = _cycle(target_loader) if supports_mmd else None

        best_val = float("inf")
        best_epoch = -1
        best_state = copy.deepcopy(self.model.state_dict())
        wait = 0
        history: Dict[str, list] = {"train_loss": [], "val_loss": [], "task": [], "mmd": [], "lambda": []}
        epoch = 0

        for epoch in range(self.epochs):
            self.model.train()
            totals = {"loss": 0.0, "task": 0.0, "mmd": 0.0}
            n_batches = 0

            for batch in train_loader:
                batch_src, batch_y = batch[0], batch[1]
                pred = self.model(batch_src)
                loss = self.supervised(pred, batch_y)
                task_value = float(loss.detach())
                mmd_value = 0.0

                if supports_mmd:
                    lam = self._current_lambda(epoch)
                    if lam > 0.0:
                        batch_tgt = next(target_stream)[0]
                        feat_src = self.model.mmd_representation(batch_src)
                        feat_tgt = self.model.mmd_representation(batch_tgt)
                        mmd = multi_kernel_mmd(feat_src, feat_tgt, **self.mmd_kwargs)
                        loss = loss + lam * mmd
                        mmd_value = float(mmd.detach())

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), float(self.grad_clip))
                self.optimizer.step()

                totals["loss"] += float(loss.detach())
                totals["task"] += task_value
                totals["mmd"] += mmd_value
                n_batches += 1

            n_batches = max(n_batches, 1)
            self.model.eval()
            val_total = 0.0
            with torch.no_grad():
                for batch_val, batch_yval in val_loader:
                    val_total += float(self.val_loss(self.model(batch_val), batch_yval))
            val_loss = val_total / max(len(val_loader), 1)
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            history["train_loss"].append(totals["loss"] / n_batches)
            history["val_loss"].append(val_loss)
            history["task"].append(totals["task"] / n_batches)
            history["mmd"].append(totals["mmd"] / n_batches)
            history["lambda"].append(self._current_lambda(epoch))

            if val_loss < best_val - 1e-12:
                best_val = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    if tag:
                        log.info("[%s] early stop at epoch %d (best %d)", tag, epoch, best_epoch)
                    break

        self.model.load_state_dict(best_state)
        return TrainResult(
            best_val_loss=best_val,
            best_epoch=best_epoch,
            epochs_run=epoch + 1,
            state_dict=best_state,
            history=history,
        )

    def _current_lambda(self, epoch: int) -> float:
        if not self.lambda_schedule:
            return self.lambda_mmd
        idx = min(epoch, len(self.lambda_schedule) - 1)
        return float(self.lambda_schedule[idx])

    # ----------------------------------------------------------- persistence
    def save(self, result: TrainResult, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "state_dict": result.state_dict,
            "best_val_loss": result.best_val_loss,
            "best_epoch": result.best_epoch,
            "epochs_run": result.epochs_run,
            "history": result.history,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> Dict[str, Any]:
    """Load a checkpoint written by :meth:`Trainer.save` into ``model``."""
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    return payload
