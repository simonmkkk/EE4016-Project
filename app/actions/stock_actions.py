from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ..paths import PROJECT_ROOT, RESULTS_DIR, SAVE_DIR
from ..services import infer_ticker_from_csv, pick_csvs, pick_model, pick_protocol, rel
from ..ui import ask, ask_yes_no


_LINE_WIDTH = 70
TRAIN_MENU_HEADER = (
    "\n" + "=" * _LINE_WIDTH + "\n"
    "Stock ML - Train Menu\n"
    "" + "=" * _LINE_WIDTH + "\n"
    "\nPlease select an option:\n\n"
    "[Train]\n"
    "   1. Train from historical data CSV(s) (pick one or more CSVs with same ticker) (train_stock.py)\n"
    "\n   0. Back\n"
    "" + "=" * _LINE_WIDTH
)

BACKTEST_MENU_HEADER = (
    "\n" + "=" * _LINE_WIDTH + "\n"
    "Stock ML - Backtest Menu\n"
    "" + "=" * _LINE_WIDTH + "\n"
    "\nPlease select an option:\n\n"
    "[Backtest]\n"
    "   1. Backtest (pick one or more CSVs + model) (backtest_stock.py)\n"
    "\n   0. Back\n"
    "" + "=" * _LINE_WIDTH
)

PREDICT_MENU_HEADER = (
    "\n" + "=" * _LINE_WIDTH + "\n"
    "Stock ML - Predict Menu\n"
    "" + "=" * _LINE_WIDTH + "\n"
    "\nPlease select an option:\n\n"
    "[Predict]\n"
    "   1. Predict + explanation (pick CSV(s) + model) (predict_stock.py)\n"
    "\n   0. Back\n"
    "" + "=" * _LINE_WIDTH
)


def _print_block(title: str, rows: list[tuple[str, str]], *, sep: str = "-") -> None:
    print("\n" + sep * _LINE_WIDTH)
    print(f"  {title}")
    print(sep * _LINE_WIDTH)
    for key, value in rows:
        print(f"  {key:<14}: {value}")
    print(sep * _LINE_WIDTH)


def _print_list(title: str, items: list[str]) -> None:
    print(f"  {title}")
    for item in items:
        print(f"    - {item}")


def run_script(script: str, extra_args: list[str] | None = None):
    extra_args = extra_args or []
    cmd = [sys.executable, script, *extra_args]
    print("\n" + "=" * _LINE_WIDTH)
    print(f"  RUN  : {script}")
    print("=" * _LINE_WIDTH)
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    print("=" * _LINE_WIDTH)
    print(f"  DONE : {script}")
    print("=" * _LINE_WIDTH + "\n")


def run_download_by_protocol():
    protocol = pick_protocol()
    if not protocol:
        return
    window_idx = ask("window_idx", "0")
    print(f"  Selected protocol: {rel(protocol)}")
    print(f"  Selected window_idx: {window_idx}")
    _print_block(
        "Download Summary",
        [
            ("Mode", "protocol"),
            ("Protocol", rel(protocol)),
            ("Window Index", window_idx),
        ],
    )
    if ask_yes_no("Ready to download?", default=True):
        print("")
        run_script("get_stock_data.py", ["--protocol", str(protocol), "--window_idx", str(window_idx)])
    else:
        print("  Cancelled.")


def run_train_from_historical_csv() -> bool:
    while True:
        csv_paths = pick_csvs(header=TRAIN_MENU_HEADER)
        if not csv_paths:
            return False

        tic = infer_ticker_from_csv(csv_paths[0])
        if not tic:
            print("  [ERROR] Cannot infer ticker from CSV name.")
            if ask_yes_no("Try again?", default=True):
                continue
            return False

        invalid = [p for p in csv_paths if infer_ticker_from_csv(p) != tic]
        if invalid:
            print("  [ERROR] Selected CSV files must all belong to the same ticker.")
            for p in invalid:
                print(f"    - {rel(p)}")
            if ask_yes_no("Select a different set of CSV files?", default=True):
                continue
            return False

        selected_csvs = [rel(p) for p in csv_paths]
        _print_list("Selected CSVs:", selected_csvs)
        print(f"  Selected ticker  : {tic}")

        epochs = ask("epochs", "100")
        print(f"  Selected epochs  : {epochs}")

        window = ask("window", "30")
        print(f"  Selected window  : {window}")

        use_attn = ask_yes_no("Use Attention?", default=True)
        print(f"  Selected attention: {'enabled' if use_attn else 'disabled'}")

        _ltq_in = ask(
            "label_threshold_quantile (per-series; Enter=0.55; type fixed for fixed threshold)",
            "0.55",
        )
        _ltq: float | None = None
        _ltq_raw = _ltq_in.strip().lower()
        if _ltq_raw in ("fixed", "manual"):
            print("  Selected label mode  : fixed threshold (next prompt)")
        elif _ltq_in.strip():
            try:
                _ltq = float(_ltq_in)
                if not (0.0 < _ltq < 1.0):
                    raise ValueError
                print(f"  Selected label mode  : per-series quantile {_ltq:.2f} (top {1-_ltq:.0%} UP per series)")
            except ValueError:
                print("  [WARN] Invalid quantile, falling back to fixed threshold.")
                _ltq = None

        if _ltq is None:
            label_threshold = ask("label_threshold (0=any up, e.g. 0.003=+0.3% only)", "0")
            try:
                _lt = float(label_threshold)
                if _lt < 0:
                    raise ValueError
            except ValueError:
                print("  [WARN] Invalid label_threshold, using 0.")
                label_threshold = "0"
                _lt = 0.0
            _lt_desc = f"log_ret > {_lt:.4f}" if _lt > 0 else "any positive return"
            print(f"  Selected label_threshold: {label_threshold} ({_lt_desc})")
        else:
            label_threshold = "0"
            _lt = 0.0
            _lt_desc = f"per-series quantile {_ltq:.2f}"

        extra = ["--csvs", *[str(p) for p in csv_paths], "--ticker", tic, "--save_model", "model.pt", "--window", str(window), "--epochs", str(epochs)]
        if use_attn:
            extra.append("--use_attn")
        if _ltq is not None:
            extra += ["--label_threshold_quantile", str(_ltq)]
        elif _lt != 0.0:
            extra += ["--label_threshold", label_threshold]
        _print_block(
            "Train Summary",
            [
                ("CSV Count", str(len(csv_paths))),
                ("Ticker", tic),
                ("Epochs", str(epochs)),
                ("Window", str(window)),
                ("Attention", "enabled" if use_attn else "disabled"),
                ("Label mode", _lt_desc),
            ],
        )
        _print_list("CSV Files:", selected_csvs)
        if ask_yes_no("Ready to train?", default=True):
            print("")
            run_script("train_stock.py", extra)
            return True
        print("  Cancelled.")
        return False


def run_backtest_pick_csv_and_model():
    list_title = (
        f"  Select CSV(s) to backtest ({rel(SAVE_DIR)}/*.csv)\n"
        "  Use comma / range (e.g. 1,3-5), or `a` / `all` for every file listed."
    )
    csv_paths = pick_csvs(
        header=BACKTEST_MENU_HEADER,
        state_key="last_backtest_csvs",
        list_title=list_title,
    )
    if not csv_paths:
        return
    tickers = {infer_ticker_from_csv(p) for p in csv_paths}
    tickers.discard(None)
    if len(tickers) != 1:
        print(
            "\n  [ERROR] All selected CSVs must be for the same ticker "
            f"(found: {', '.join(sorted(tickers)) or 'unknown'})."
        )
        return
    tic = next(iter(tickers))
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

    threshold_prompt = (
        f"decision_threshold (blank = use metadata eval_threshold={meta_threshold:g})"
        if isinstance(meta_threshold, float)
        else "decision_threshold (blank = use metadata eval_threshold)"
    )
    decision_threshold = ask(threshold_prompt, "")
    selected_threshold = (
        decision_threshold
        if decision_threshold
        else f"metadata default ({meta_threshold:g})" if isinstance(meta_threshold, float) else "metadata default"
    )

    transaction_fee_rate = ask("transaction_fee_rate", "0.001")
    evaluation_split = ask("evaluation_split (test/all)", "test").lower()

    attach_protocol_baseline_parameters = ask_yes_no("attach_protocol_baseline_parameters?", default=False)
    protocol = pick_protocol() if attach_protocol_baseline_parameters else None

    print("\n  Backtest Selections")
    print(f"  csv_path(s)              : {len(csv_paths)} file(s)")
    for p in csv_paths:
        print(f"    - {rel(p)}")
    print(f"  model_path               : {rel(model_path)}")
    print(f"  decision_threshold       : {selected_threshold}")
    print(f"  transaction_fee_rate     : {transaction_fee_rate}")
    print(f"  evaluation_split         : {evaluation_split}")
    if protocol is not None:
        print(f"  protocol_path            : {rel(protocol)}")
    else:
        print("  protocol_path            : none")

    extra_base = [
        "--model",
        str(model_path),
        "--fee",
        str(transaction_fee_rate),
        "--eval_split",
        evaluation_split,
    ]
    if decision_threshold:
        extra_base += ["--threshold", decision_threshold]
    if protocol is not None:
        extra_base += ["--protocol", str(protocol)]

    print("\n  Backtest Summary")
    print(f"  csv count                : {len(csv_paths)}")
    print(f"  model_path               : {rel(model_path)}")
    print(f"  evaluation_split         : {evaluation_split}")
    print(f"  transaction_fee_rate     : {transaction_fee_rate}")
    print(f"  decision_threshold       : {selected_threshold}")
    print(f"  protocol_path            : {rel(protocol) if protocol is not None else 'none'}")
    if ask_yes_no("Ready to backtest?", default=True):
        print("")
        recap_rows: list[tuple[str, float, float, float]] = []
        for csv_path in csv_paths:
            extra = ["--csv", str(csv_path), *extra_base]
            try:
                run_script("backtest_stock.py", extra)
            except subprocess.CalledProcessError as e:
                print(f"\n  [WARN] backtest_stock.py failed for {rel(csv_path)} (exit {e.returncode}); continuing.\n")
                stem = csv_path.stem
                recap_rows.append((stem, float("nan"), float("nan"), float("nan")))
                continue
            stem = csv_path.stem
            summary_path = RESULTS_DIR / tic / f"{stem}_bt_summary.json"
            if summary_path.exists():
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        s = json.load(f)
                    if s.get("status") == "skipped":
                        recap_rows.append((stem, float("nan"), float("nan"), float("nan")))
                        continue
                    ml = s.get("strategies", {}).get("model_lstm", {})
                    pm = s.get("predictive_metrics_model", {})
                    recap_rows.append(
                        (
                            stem,
                            float(ml.get("total_return", 0.0)),
                            float(pm.get("accuracy", 0.0)),
                            float(pm.get("f1", 0.0)),
                        )
                    )
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    recap_rows.append((stem, float("nan"), float("nan"), float("nan")))
            else:
                recap_rows.append((stem, float("nan"), float("nan"), float("nan")))

        if len(recap_rows) > 1:
            print("\n" + "=" * _LINE_WIDTH)
            print("  BACKTEST RECAP (all runs)")
            print("=" * _LINE_WIDTH)
            print(f"  {'CSV stem':<36}  {'lstm_ret':>10}  {'accuracy':>10}  {'f1':>8}")
            print("  " + "-" * 66)
            for stem, ret, acc, f1 in recap_rows:
                if ret != ret:
                    print(f"  {stem:<36}  {'(n/a)':>10}  {'(n/a)':>10}  {'(n/a)':>8}")
                else:
                    print(f"  {stem:<36}  {ret:>10.4f}  {acc:>10.4f}  {f1:>8.4f}")
            print("=" * _LINE_WIDTH + "\n")
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
    _print_block(
        "Pipeline Summary",
        [
            ("Protocol", rel(protocol)),
            ("Epochs", str(epochs)),
            ("Window", str(window)),
            ("Fee", str(fee)),
        ],
    )
    if ask_yes_no("Ready to run pipeline?", default=True):
        print("")
        run_script("run_protocol.py", ["--protocol", str(protocol), "--epochs", str(epochs), "--window", str(window), "--fee", str(fee)])
    else:
        print("  Cancelled.")


def run_predict_pick_csv_and_model():
    list_title = (
        f"  Select CSV(s) for predict ({rel(SAVE_DIR)}/*.csv)\n"
        "  Use comma / range (e.g. 1,3-5), or `a` / `all` for every file listed."
    )
    csv_paths = pick_csvs(
        header=PREDICT_MENU_HEADER,
        state_key="last_pred_csv",
        list_title=list_title,
    )
    if not csv_paths:
        return
    tickers = {infer_ticker_from_csv(p) for p in csv_paths}
    tickers.discard(None)
    if len(tickers) != 1:
        print(
            "\n  [ERROR] All selected CSVs must be for the same ticker "
            f"(found: {', '.join(sorted(tickers)) or 'unknown'})."
        )
        return
    tic = next(iter(tickers))
    model_path = pick_model(prefer_ticker=tic, state_key="last_pred_model")
    if not model_path:
        return

    print("\n  Predict Selections")
    print(f"  csv_path(s)              : {len(csv_paths)} file(s)")
    for p in csv_paths:
        print(f"    - {rel(p)}")
    print(f"  model_path               : {rel(model_path)}")

    _print_block(
        "Predict Summary",
        [
            ("CSV count", str(len(csv_paths))),
            ("Ticker", tic),
            ("Model", rel(model_path)),
        ],
    )
    _print_list("CSV files:", [rel(p) for p in csv_paths])

    if ask_yes_no("Ready to predict?", default=True):
        print("")
        for csv_path in csv_paths:
            try:
                run_script("predict_stock.py", ["--csv", str(csv_path), "--model", str(model_path)])
            except subprocess.CalledProcessError as e:
                print(
                    f"\n  [WARN] predict_stock.py failed for {rel(csv_path)} "
                    f"(exit {e.returncode}); continuing.\n"
                )
    else:
        print("  Cancelled.")

