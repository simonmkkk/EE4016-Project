from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ..paths import PROJECT_ROOT
from ..services import infer_ticker_from_csv, pick_csv, pick_model, pick_protocol, rel
from ..ui import ask, ask_yes_no


def run_script(script: str, extra_args: list[str] | None = None):
    extra_args = extra_args or []
    cmd = [sys.executable, script, *extra_args]
    print("\n" + "=" * 70)
    print(f"  Running: {script}")
    print("=" * 70)
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    print("=" * 70)
    print(f"  Finished: {script}")
    print("=" * 70 + "\n")


def run_download_by_protocol():
    protocol = pick_protocol()
    if not protocol:
        return
    window_idx = ask("window_idx", "0")
    print(f"  Selected protocol: {rel(protocol)}")
    print(f"  Selected window_idx: {window_idx}")
    print("\n" + "-" * 70)
    print("  Download Summary")
    print("-" * 70)
    print(f"  Mode           : protocol")
    print(f"  Protocol       : {rel(protocol)}")
    print(f"  Window Index   : {window_idx}")
    print("-" * 70)
    if ask_yes_no("Ready to download?", default=True):
        print("")
        run_script("get_stock_data.py", ["--protocol", str(protocol), "--window_idx", str(window_idx)])
    else:
        print("  Cancelled.")


def run_train_from_historical_csv():
    csv_path = pick_csv()
    if not csv_path:
        return
    tic = infer_ticker_from_csv(csv_path)
    if not tic:
        print("  [ERROR] Cannot infer ticker from CSV name.")
        return
    print(f"  Selected CSV: {rel(csv_path)}")
    print(f"  Selected ticker: {tic}")
    print("")

    epochs = ask("epochs", "5")
    print(f"  Selected epochs: {epochs}")
    print("")

    window = ask("window", "30")
    print(f"  Selected window: {window}")
    print("")

    use_attn = ask_yes_no("Use Attention?", default=True)
    print(f"  Selected attention: {'yes' if use_attn else 'no'}")

    extra = [
        "--csv_dir",
        "historical_data",
        "--ticker",
        tic,
        "--save_model",
        "dir_model.pt",
        "--window",
        str(window),
        "--epochs",
        str(epochs),
    ]
    if use_attn:
        extra.append("--use_attn")
    print("\n" + "-" * 70)
    print("  Train Summary")
    print("-" * 70)
    print(f"  CSV Dir        : historical_data")
    print(f"  Ticker         : {tic}")
    print(f"  Epochs         : {epochs}")
    print(f"  Window         : {window}")
    print(f"  Attention      : {'enabled' if use_attn else 'disabled'}")
    print("-" * 70)
    if ask_yes_no("Ready to train?", default=True):
        print("")
        run_script("train_stock.py", extra)
    else:
        print("  Cancelled.")


def run_backtest_pick_csv_and_model():
    csv_path = pick_csv()
    if not csv_path:
        return
    tic = infer_ticker_from_csv(csv_path)
    model_path = pick_model(prefer_ticker=tic)
    if not model_path:
        return

    meta_threshold = None
    try:
        meta_path = Path(model_path).with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_obj = json.load(f)
            val = meta_obj.get("eval_threshold")
            if isinstance(val, (int, float)):
                meta_threshold = float(val)
    except Exception:
        meta_threshold = None

    thr_hint = (
        f"threshold (blank=use meta eval_threshold={meta_threshold:g})"
        if isinstance(meta_threshold, float)
        else "threshold (blank=use meta eval_threshold)"
    )
    threshold = ask(thr_hint, "")
    print(f"  Selected threshold: {threshold if threshold else f'meta default ({meta_threshold:g})' if isinstance(meta_threshold, float) else 'meta default'}")
    print("")

    fee = ask("transaction fee", "0.001")
    print(f"  Selected transaction fee: {fee}")
    print("")

    eval_split = ask("eval_split (test/all)", "test").lower()
    print(f"  Selected eval_split: {eval_split}")
    print("")

    use_protocol = ask_yes_no("Attach protocol baseline_params?", default=False)
    print(f"  Selected attach protocol baseline_params: {'yes' if use_protocol else 'no'}")
    print("")
    protocol = pick_protocol() if use_protocol else None

    print(f"  Selected CSV: {rel(csv_path)}")
    print(f"  Selected model: {rel(model_path)}")
    if protocol is not None:
        print(f"  Selected protocol: {rel(protocol)}")
    else:
        print("  Selected protocol: none")

    extra = ["--csv", str(csv_path), "--model", str(model_path), "--fee", str(fee), "--eval_split", eval_split]
    if threshold:
        extra += ["--threshold", threshold]
    if protocol is not None:
        extra += ["--protocol", str(protocol)]
    print("\n" + "-" * 70)
    print("  Backtest Summary")
    print("-" * 70)
    print(f"  CSV            : {rel(csv_path)}")
    print(f"  Model          : {rel(model_path)}")
    print(f"  Eval Split     : {eval_split}")
    print(f"  Transaction Fee: {fee}")
    print(f"  Threshold      : {threshold if threshold else 'meta default'}")
    print(f"  Protocol       : {rel(protocol) if protocol is not None else 'none'}")
    print("-" * 70)
    if ask_yes_no("Ready to backtest?", default=True):
        print("")
        run_script("backtest_stock.py", extra)
    else:
        print("  Cancelled.")


def run_protocol_runner():
    protocol = pick_protocol()
    if not protocol:
        return
    epochs = ask("epochs", "5")
    window = ask("window", "30")
    fee = ask("fee", "0.001")
    print(f"  Selected protocol: {rel(protocol)}")
    print(f"  Selected epochs: {epochs}")
    print(f"  Selected window: {window}")
    print(f"  Selected fee: {fee}")
    print("\n" + "-" * 70)
    print("  Pipeline Summary")
    print("-" * 70)
    print(f"  Protocol       : {rel(protocol)}")
    print(f"  Epochs         : {epochs}")
    print(f"  Window         : {window}")
    print(f"  Fee            : {fee}")
    print("-" * 70)
    if ask_yes_no("Ready to run pipeline?", default=True):
        print("")
        run_script("run_protocol.py", ["--protocol", str(protocol), "--epochs", str(epochs), "--window", str(window), "--fee", str(fee)])
    else:
        print("  Cancelled.")


def run_predict_pick_csv_and_model():
    csv_path = pick_csv(state_key="last_pred_csv")
    if not csv_path:
        return
    tic = infer_ticker_from_csv(csv_path)
    model_path = pick_model(prefer_ticker=tic, state_key="last_pred_model")
    if not model_path:
        return
    print(f"  Selected CSV: {rel(csv_path)}")
    print(f"  Selected model: {rel(model_path)}")
    print("\n" + "-" * 70)
    print("  Predict Summary")
    print("-" * 70)
    print(f"  CSV            : {rel(csv_path)}")
    print(f"  Model          : {rel(model_path)}")
    print("-" * 70)
    if ask_yes_no("Ready to predict?", default=True):
        print("")
        run_script("predict_stock.py", ["--csv", str(csv_path), "--model", str(model_path)])
    else:
        print("  Cancelled.")

