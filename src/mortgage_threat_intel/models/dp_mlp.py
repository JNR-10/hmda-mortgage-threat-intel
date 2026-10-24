"""Matched non-private / DP-SGD MLP pair (J-06). CPU-first Opacus path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mortgage_threat_intel.config import PrivacyConfig
from mortgage_threat_intel.privacy.accounting import PrivacyMetadata, extract_accountant_metadata
from mortgage_threat_intel.seeding import seed_everything


class MLPBinary(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainedMLP:
    name: str
    model: MLPBinary
    device: str
    privacy: PrivacyMetadata | None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        device = torch.device(self.device)
        with torch.no_grad():
            xt = torch.as_tensor(X, dtype=torch.float32, device=device)
            logits = self.model(xt)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba


def _train_loop(
    model: MLPBinary,
    loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    privacy_engine: Any | None = None,
) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if privacy_engine is not None:
        # Opacus already wrapped optimizer/model/loader by caller
        pass
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()


def train_matched_mlp_pair(
    X: np.ndarray,
    y: np.ndarray,
    privacy_cfg: PrivacyConfig,
    *,
    seed: int,
    device: str = "cpu",
) -> tuple[TrainedMLP, TrainedMLP]:
    """Train non-private MLP and matched DP-SGD MLP on the same data/architecture."""
    seed_everything(seed)
    device_t = torch.device(device if device != "mps" else "cpu")  # DP path CPU-first
    if device == "mps":
        # Plan: add MPS only after dedicated Opacus smoke; force CPU for DP pair.
        device_t = torch.device("cpu")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    n = len(y)
    in_dim = X.shape[1]
    hidden = list(privacy_cfg.mlp.hidden_dims)
    epochs = int(privacy_cfg.mlp.epochs)
    batch_size = int(privacy_cfg.mlp.batch_size)
    lr = float(privacy_cfg.mlp.learning_rate)
    wd = float(privacy_cfg.mlp.weight_decay)
    dropout = float(privacy_cfg.mlp.dropout)

    def _loader(xs: np.ndarray, ys: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.as_tensor(xs, dtype=torch.float32),
            torch.as_tensor(ys, dtype=torch.long),
        )
        return DataLoader(ds, batch_size=min(batch_size, max(len(ys), 1)), shuffle=shuffle)

    # Non-private control
    seed_everything(seed)
    nonpriv_model = MLPBinary(in_dim, hidden, dropout=dropout).to(device_t)
    _train_loop(
        nonpriv_model,
        _loader(X, y, shuffle=True),
        epochs=epochs,
        lr=lr,
        weight_decay=wd,
        device=device_t,
    )
    nonpriv = TrainedMLP(
        name="mlp_nonprivate", model=nonpriv_model, device=str(device_t), privacy=None
    )

    # DP-SGD target
    seed_everything(seed)
    dp_model = MLPBinary(in_dim, hidden, dropout=dropout).to(device_t)
    delta = 1.0 / max(n, 1)
    target_eps = float(privacy_cfg.dp_sgd.target_epsilon)
    max_grad_norm = float(privacy_cfg.dp_sgd.max_grad_norm)
    noise_multiplier = privacy_cfg.dp_sgd.noise_multiplier

    from opacus import PrivacyEngine

    train_loader = _loader(X, y, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(dp_model.parameters(), lr=lr, weight_decay=wd)
    privacy_engine = PrivacyEngine()

    # Make sample rate / steps explicit for accountant metadata
    sample_rate = min(batch_size, n) / max(n, 1)
    steps_per_epoch = int(np.ceil(n / max(min(batch_size, n), 1)))
    expected_steps = steps_per_epoch * epochs

    if noise_multiplier is None:
        # Let Opacus make private with target epsilon (RDP accountant)
        dp_model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=dp_model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=epochs,
            target_epsilon=target_eps,
            target_delta=delta,
            max_grad_norm=max_grad_norm,
        )
    else:
        dp_model, optimizer, train_loader = privacy_engine.make_private(
            module=dp_model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=float(noise_multiplier),
            max_grad_norm=max_grad_norm,
        )

    # Prefer optimizer attribute; fall back to config / NaN for manifests
    resolved_noise = getattr(optimizer, "noise_multiplier", None)
    if resolved_noise is None:
        resolved_noise = noise_multiplier
    noise_multiplier = float(resolved_noise) if resolved_noise is not None else float("nan")

    dp_model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb = xb.to(device_t)
            yb = yb.to(device_t)
            optimizer.zero_grad()
            logits = dp_model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    achieved_eps = float(privacy_engine.get_epsilon(delta))
    meta = extract_accountant_metadata(
        target_epsilon=target_eps,
        achieved_epsilon=achieved_eps,
        delta=delta,
        sample_rate=float(sample_rate),
        steps=int(expected_steps),
        noise_multiplier=float(noise_multiplier),
        max_grad_norm=max_grad_norm,
        n_train=n,
        adjacency=privacy_cfg.dp_sgd.adjacency,
        device=str(device_t),
        accountant_name=type(privacy_engine.accountant).__name__,
    )
    dpriv = TrainedMLP(name="mlp_dp_sgd", model=dp_model, device=str(device_t), privacy=meta)
    return nonpriv, dpriv
