"""Windowed tensors for LSTM train / eval / inference."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def build_seq(
    frame: pd.DataFrame,
    feats_lstm: list[str],
    window: int,
    *,
    use_interval_embedding: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Training-style sequences: label at index i uses rows [i-window, i)."""
    v = frame[feats_lstm].values.astype(np.float32)
    X, y, iv = [], [], []
    for i in range(window, len(frame)):
        X.append(v[i - window : i])
        y.append(d[i])
        if use_interval_embedding:
            iv.append(int(frame["interval_id"].iloc[i]))
    if not X:
        d_feat = len(feats_lstm)
        empty_y = np.empty((0, 1), dtype=np.float32)
        if use_interval_embedding:
            return (
                np.empty((0, window, d_feat), dtype=np.float32),
                empty_y,
                np.empty((0,), dtype=np.int64),
            )
        return np.empty((0, window, d_feat), dtype=np.float32), empty_y, None
    Xa = np.array(X, dtype=np.float32)
    ya = np.array(y, dtype=np.float32).reshape(-1, 1)
    if use_interval_embedding:
        return Xa, ya, np.array(iv, dtype=np.int64)
    return Xa, ya, None


def build_seq_multi(
    frames: list[pd.DataFrame],
    feats_lstm: list[str],
    window: int,
    *,
    use_interval_embedding: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    xs, ys, ivs = [], [], []
    for fr in frames:
        X_part, y_part, iv_part = build_seq(
            fr, feats_lstm, window, use_interval_embedding=use_interval_embedding
        )
        if X_part.shape[0] > 0:
            xs.append(X_part)
            ys.append(y_part)
            if iv_part is not None:
                ivs.append(iv_part)
    if not xs:
        d_feat = len(feats_lstm)
        empty_y = np.empty((0, 1), dtype=np.float32)
        if use_interval_embedding:
            return (
                np.empty((0, window, d_feat), dtype=np.float32),
                empty_y,
                np.empty((0,), dtype=np.int64),
            )
        return np.empty((0, window, d_feat), dtype=np.float32), empty_y, None
    X_cat = np.concatenate(xs, axis=0)
    y_cat = np.concatenate(ys, axis=0)
    if use_interval_embedding:
        return X_cat, y_cat, np.concatenate(ivs, axis=0)
    return X_cat, y_cat, None


def build_seq_x_only(
    frame: pd.DataFrame,
    feats_lstm: list[str],
    window: int,
    *,
    use_interval_embedding: bool,
    extra_next_bar: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Inference / backtest: same windows as training, plus optional last-bar window
    for forecasting the move after the latest close.
    """
    v = frame[feats_lstm].values.astype(np.float32)
    X, iv = [], []
    for i in range(window, len(frame)):
        X.append(v[i - window : i])
        if use_interval_embedding:
            iv.append(int(frame["interval_id"].iloc[i]))
    Xa = np.array(X, dtype=np.float32) if X else np.empty((0, window, len(feats_lstm)), dtype=np.float32)
    iva = np.array(iv, dtype=np.int64) if (use_interval_embedding and iv) else None

    x_next = None
    iv_next = None
    if extra_next_bar and len(frame) >= window:
        x_next = v[len(frame) - window : len(frame)].reshape(1, window, -1).astype(np.float32)
        if use_interval_embedding:
            iv_next = np.array([int(frame["interval_id"].iloc[-1])], dtype=np.int64)

    return Xa, iva, x_next, iv_next


def evaluate_split(
    model: nn.Module,
    frame: pd.DataFrame | list[pd.DataFrame],
    feats_lstm: list[str],
    window: int,
    device: torch.device,
    threshold: float,
    *,
    use_interval_embedding: bool,
) -> dict | None:
    frames = [frame] if isinstance(frame, pd.DataFrame) else frame
    X_eval, y_eval, iv_eval = build_seq_multi(
        frames, feats_lstm, window, use_interval_embedding=use_interval_embedding
    )
    if X_eval.shape[0] == 0:
        return None
    model.eval()
    x_t = torch.tensor(X_eval).to(device)
    with torch.no_grad():
        if iv_eval is not None:
            logits = model(x_t, torch.tensor(iv_eval).to(device))
        else:
            logits = model(x_t)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
    y_true = y_eval.reshape(-1).astype(int)
    y_pred = (probs > threshold).astype(int)
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }
