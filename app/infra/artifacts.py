from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..model import LSTMDir
from .files import load_json, load_pickle


@dataclass(frozen=True)
class ArtifactBundle:
    model_path: Path
    meta_path: Path
    scaler_path: Path
    meta: dict[str, Any]
    scaler: Any


def resolve_artifact_paths(
    model_path: str | Path,
    *,
    meta_path: str | Path | None = None,
    scaler_path: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    model = Path(model_path)
    meta = Path(meta_path) if meta_path else model.with_suffix(".meta.json")
    scaler = Path(scaler_path) if scaler_path else model.with_suffix(".scaler.pkl")
    return model, meta, scaler


def load_artifact_bundle(
    model_path: str | Path,
    *,
    meta_path: str | Path | None = None,
    scaler_path: str | Path | None = None,
) -> ArtifactBundle:
    model, meta, scaler = resolve_artifact_paths(
        model_path,
        meta_path=meta_path,
        scaler_path=scaler_path,
    )
    return ArtifactBundle(
        model_path=model,
        meta_path=meta,
        scaler_path=scaler,
        meta=load_json(meta),
        scaler=load_pickle(scaler),
    )


def scaled_feature_columns(
    meta: dict[str, Any],
    trained_features: list[str],
) -> list[str]:
    scaled = meta.get("features_scaled")
    if isinstance(scaled, list):
        return scaled
    if "interval_id" in trained_features:
        return [c for c in trained_features if c != "interval_id"]
    return list(trained_features)


def build_model_from_meta(
    feature_count: int,
    meta: dict[str, Any],
    *,
    device: torch.device,
    use_attn: bool | None = None,
) -> LSTMDir:
    use_interval_embedding = bool(meta.get("use_interval_embedding", False))
    num_intervals = meta.get("num_interval_embeddings")
    return LSTMDir(
        feature_count,
        att=bool(meta.get("use_attn", False) if use_attn is None else use_attn),
        num_layers=int(meta.get("num_layers", 1)),
        num_intervals=(
            int(num_intervals)
            if use_interval_embedding and num_intervals is not None
            else None
        ),
        embed_dim=int(meta.get("interval_embed_dim", 8)),
        dropout=float(meta.get("dropout", 0.3)),
    ).to(device)
