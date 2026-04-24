from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..infra.files import load_json
from ..paths import MODEL_DIR, RESULTS_DIR, SAVE_DIR
from ..reporting.merged_results import (
    flatten_backtest_summary,
    flatten_train_meta,
    read_prediction_summary,
)
from .predict_stock import main as predict_main


def run_merge() -> int:
    out_csv = RESULTS_DIR / "all_results_merged.csv"
    summaries = sorted(RESULTS_DIR.glob("*/*_bt_summary.json"))
    if not summaries:
        raise SystemExit(f"No *_bt_summary.json under {RESULTS_DIR}")

    rows: list[dict] = []
    for summary_path in summaries:
        stem = summary_path.name.replace("_bt_summary.json", "")
        ticker = stem.split("_")[0].upper()
        csv_path = SAVE_DIR / f"{stem}.csv"
        meta_path = MODEL_DIR / ticker / "model.meta.json"
        pred_path = RESULTS_DIR / ticker / f"{stem}_pred.csv"
        model_path = MODEL_DIR / ticker / "model.pt"

        backtest_summary = load_json(summary_path)
        train_meta = load_json(meta_path) if meta_path.exists() else {}

        if not pred_path.exists() and csv_path.exists() and model_path.exists():
            print(f"[INFO] missing pred for {stem}, generating...")
            try:
                predict_main(["--csv", str(csv_path), "--model", str(model_path)])
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 1
                if code != 0:
                    print(f"[WARN] predict_stock failed (skip pred) for {csv_path.name}")

        row: dict = {
            "data_stem": stem,
            "ticker": ticker,
            "bt_summary_path": str(summary_path),
            "historical_csv": str(csv_path) if csv_path.exists() else "",
        }
        row.update(flatten_train_meta(train_meta))
        row.update(flatten_backtest_summary(backtest_summary))
        row.update(read_prediction_summary(pred_path))
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] {len(df)} rows -> {out_csv}")
    return 0


def main(argv: list[str] | None = None) -> int:
    _ = argv
    return run_merge()


if __name__ == "__main__":
    raise SystemExit(main())
