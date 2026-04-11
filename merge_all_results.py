#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Merge existing train (model.meta.json), backtest (*_bt_summary.json), predict (*_pred.csv) into one CSV."""
from __future__ import annotations

import json
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from app.paths import MODEL_DIR, RESULTS_DIR, SAVE_DIR


def _flatten_train_meta(meta: dict) -> dict:
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
    for k, v in meta.items():
        if k in skip:
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[f"train_{k}"] = v
    for kind in ("split_sizes_rows", "split_sizes_sequences"):
        block = meta.get(kind)
        if isinstance(block, dict):
            for sk, sv in block.items():
                out[f"train_{kind}_{sk}"] = sv
    m = meta.get("metrics") or {}
    for split in ("val", "test"):
        mm = m.get(split)
        if not isinstance(mm, dict):
            continue
        for mk, mv in mm.items():
            out[f"train_metrics_{split}_{mk}"] = mv
    wf = m.get("walk_forward")
    if isinstance(wf, list) and wf:
        out["train_metrics_walk_forward_json"] = json.dumps(wf, ensure_ascii=False)
    return out


def _flatten_bt_summary(s: dict) -> dict:
    out: dict = {}
    if not s:
        return out
    out["bt_threshold"] = s.get("threshold")
    out["bt_fee"] = s.get("fee")
    out["bt_eval_split"] = s.get("eval_split")
    predm = s.get("predictive_metrics_model") or {}
    out["bt_predictive_accuracy"] = predm.get("accuracy")
    out["bt_predictive_f1"] = predm.get("f1")
    for strat, vals in (s.get("strategies") or {}).items():
        if not isinstance(vals, dict):
            continue
        for k, v in vals.items():
            out[f"bt_{strat}_{k}"] = v
    return out


def _read_pred_summary(pred_path: Path) -> dict:
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
    dfp = pd.read_csv(StringIO("\n".join(body)))
    if "correct" in dfp.columns:
        out["pred_accuracy"] = float(dfp["correct"].astype(bool).mean())
        out["pred_n_rows"] = int(len(dfp))
    if "high_conf_wrong" in dfp.columns:
        out["pred_high_conf_wrong"] = int(dfp["high_conf_wrong"].astype(bool).sum())
    return out


def _run_predict(csv_path: Path, model_path: Path) -> bool:
    cmd = [
        sys.executable,
        str(_ROOT / "predict_stock.py"),
        "--csv",
        str(csv_path),
        "--model",
        str(model_path),
    ]
    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(_ROOT))
    if proc.returncode != 0:
        print(f"[WARN] predict_stock failed (skip pred) for {csv_path.name}")
        return False
    return True


def main():
    out_csv = RESULTS_DIR / "all_results_merged.csv"
    summaries = sorted(RESULTS_DIR.glob("*/*_bt_summary.json"))
    if not summaries:
        print(f"No *_bt_summary.json under {RESULTS_DIR}")
        sys.exit(1)

    rows: list[dict] = []
    for summary_path in summaries:
        stem = summary_path.name.replace("_bt_summary.json", "")
        ticker_dir = summary_path.parent.name.upper()
        ticker = stem.split("_")[0].upper()

        csv_path = SAVE_DIR / f"{stem}.csv"
        meta_path = MODEL_DIR / ticker / "model.meta.json"
        pred_path = RESULTS_DIR / ticker / f"{stem}_pred.csv"

        with open(summary_path, "r", encoding="utf-8") as f:
            bt = json.load(f)

        meta: dict = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        model_path = MODEL_DIR / ticker / "model.pt"
        if not pred_path.exists() and csv_path.exists() and model_path.exists():
            print(f"[INFO] missing pred for {stem}, generating…")
            _run_predict(csv_path, model_path)

        row: dict = {
            "data_stem": stem,
            "ticker": ticker,
            "bt_summary_path": str(summary_path.relative_to(_ROOT)),
            "historical_csv": str(csv_path.relative_to(_ROOT)) if csv_path.exists() else "",
        }
        row.update(_flatten_train_meta(meta))
        row.update(_flatten_bt_summary(bt))
        row.update(_read_pred_summary(pred_path))
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] {len(df)} rows -> {out_csv.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
