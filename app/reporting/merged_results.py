from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pandas as pd


def flatten_train_meta(meta: dict) -> dict:
    out: dict = {}
    if not meta:
        return out
    skip = {
        "metrics",
        "data_fingerprints",
        "features",
        "features_scaled",
        "interval_id_map",
        "reproducibility",
        "protocol",
    }
    for key, value in meta.items():
        if key in skip:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[f"train_{key}"] = value
    for kind in ("split_sizes_rows", "split_sizes_sequences"):
        block = meta.get(kind)
        if isinstance(block, dict):
            for sub_key, sub_value in block.items():
                out[f"train_{kind}_{sub_key}"] = sub_value
    metrics = meta.get("metrics") or {}
    for split in ("val", "test"):
        split_metrics = metrics.get(split)
        if not isinstance(split_metrics, dict):
            continue
        for metric_key, metric_value in split_metrics.items():
            out[f"train_metrics_{split}_{metric_key}"] = metric_value
    walk_forward = metrics.get("walk_forward")
    if isinstance(walk_forward, list) and walk_forward:
        out["train_metrics_walk_forward_json"] = json.dumps(
            walk_forward,
            ensure_ascii=False,
        )
    return out


def flatten_backtest_summary(summary: dict) -> dict:
    out: dict = {}
    if not summary:
        return out
    out["bt_threshold"] = summary.get("threshold")
    out["bt_fee"] = summary.get("fee")
    out["bt_eval_split"] = summary.get("eval_split")
    predictive_metrics = summary.get("predictive_metrics_model") or {}
    out["bt_predictive_accuracy"] = predictive_metrics.get("accuracy")
    out["bt_predictive_f1"] = predictive_metrics.get("f1")
    for strategy, values in (summary.get("strategies") or {}).items():
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            out[f"bt_{strategy}_{key}"] = value
    return out


def read_prediction_summary(pred_path: Path) -> dict:
    out: dict = {}
    if not pred_path.exists():
        out["pred_present"] = False
        return out
    out["pred_present"] = True
    text = pred_path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    body: list[str] = []
    for line in lines:
        if line.strip().lower().startswith("accuracy"):
            parts = line.split(",")
            out["pred_accuracy_footer"] = parts[-1].strip() if parts else None
            continue
        if line.strip():
            body.append(line)
    if not body:
        return out
    df = pd.read_csv(StringIO("\n".join(body)))
    if "correct" in df.columns:
        out["pred_accuracy"] = float(df["correct"].astype(bool).mean())
        out["pred_n_rows"] = int(len(df))
    if "high_conf_wrong" in df.columns:
        out["pred_high_conf_wrong"] = int(df["high_conf_wrong"].astype(bool).sum())
    return out
