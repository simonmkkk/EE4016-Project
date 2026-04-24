# Adaptive Multi-Granularity Neural Trading System

This project implements an end-to-end stock prediction and evaluation pipeline built around an LSTM-based direction classifier. It supports:

1. Historical data download with `get_stock_data.py`
2. Model training with `train_stock.py`
3. Standalone prediction with `predict_stock.py`
4. Backtesting with `backtest_stock.py`
5. Protocol-driven batch experiments with `run_protocol.py`

## Flow Diagram

The editable diagram source is [flowchart/flowchart.drawio](flowchart/flowchart.drawio). It describes the full pipeline coordinated by `experiment_protocol.json` and `run_protocol.py`, from data download to training, prediction, backtesting, and summary generation.

Training writes reproducible artifacts under `model/<SYMBOL>/`, including the model weights, scaler, and metadata. Prediction can be run independently and outputs explanation-oriented columns. Backtesting compares the LSTM strategy against benchmark strategies on the same evaluation segment and writes outputs to `RESULTS_DIR` (default: `backtest_results/`, configured in `.env`).

The interactive entry point is `main.py`. Run `uv run main.py` from the project root to open the menu-driven interface.

## Project Structure

```text
EE4016-Project
|
+-- main.py
+-- get_stock_data.py
+-- train_stock.py
+-- predict_stock.py
+-- backtest_stock.py
+-- run_protocol.py
+-- merge_all_results.py
+-- experiment_protocol.json
+-- pyproject.toml
+-- README.md
+-- flowchart/
|   +-- flowchart.drawio
|   `-- flowchart.png
|
+-- app/
|   +-- menu.py
|   +-- constants.py
|   +-- fe.py
|   +-- model.py
|   +-- paths.py
|   +-- sequences.py
|   +-- state.py
|   +-- actions/
|   |   `-- stock_actions.py
|   +-- menus/
|   |   +-- data_menu.py
|   |   +-- train_menu.py
|   |   +-- backtest_menu.py
|   |   +-- predict_menu.py
|   |   +-- pipeline_menu.py
|   |   `-- results_menu.py
|   +-- services/
|   |   `-- discovery.py
|   `-- ui/
|       `-- cli.py
|
+-- historical_data/    (or `SAVE_DIR` from `.env`)
+-- model/              (or `MODEL_DIR` from `.env`)
`-- backtest_results/   (or `RESULTS_DIR` from `.env`)
```

## Using `main.py`

`main.py` is the interactive launcher. It does not parse command-line options. After startup, it enters the main menu loop through `app.menu.interactive_menu`, clears the terminal, prints the menu, and waits for numeric input.

Before launching it:

1. Run `uv sync` in the project root.
2. Configure `.env` as described below.
3. Make sure `SAVE_DIR`, `MODEL_DIR`, and `RESULTS_DIR` point to valid folders.

The menu scans `SAVE_DIR` for `*.csv` files that can be selected for training, prediction, and backtesting.

Start it from the project root:

```powershell
uv run main.py
```

### Main Menu Options

| Input | Section  | Action |
| ---- | -------- | ------ |
| `1` | Data | Download data by ticker, interval, and lookback range using `get_stock_data.py` |
| `2` | Train | Select historical CSV files and train a model with `train_stock.py` |
| `3` | Backtest | Select CSV files and a model, then run `backtest_stock.py` |
| `4` | Predict | Select CSV files and a model, then run `predict_stock.py` |
| `5` | Pipeline | Run the protocol batch runner via `run_protocol.py` |
| `6` | Results | Show a summary of `RESULTS_DIR/comparison_summary.csv` |
| `0` | Exit | Quit the program |

Invalid input will prompt for a new selection. Pressing `Ctrl+C` or sending EOF exits the program cleanly. Most submenus use `0` to return to the previous level.

### Data Download Call Flow

```text
main.py
  -> app/menu.py
  -> app/menus/data_menu.py
  -> app/actions/stock_actions.py::run_script(...)
  -> subprocess.run([sys.executable, "get_stock_data.py", ...])
  -> get_stock_data.py
  -> <SAVE_DIR>/<TICKER>_<INTERVAL>_<LOOKBACK>.csv
```

## 1. Environment Requirements

- Windows, macOS, or Linux
- Python 3.12+
- `uv`

Install dependencies in the project root:

```powershell
uv sync
```

This creates `.venv` and installs the required packages.

### PyTorch: CUDA 12.4 and CPU Note

`pyproject.toml` currently pins `torch==2.6.0+cu124` and `torchvision==0.21.0+cu124`, using the PyTorch CUDA 12.4 wheel index through `tool.uv.sources`.

If the target machine does not have a compatible NVIDIA driver, CUDA runtime, or access to the PyTorch CUDA index, `uv sync` may fail. In that case, switch to CPU wheels using the official PyTorch installation guide and update `pyproject.toml` accordingly before syncing again.

Model artifacts such as `model.pt` are not tied to a specific GPU, but inference still requires a compatible Python environment and dependency set.

## 2. Configure `.env` First

Copy `.env_example` to `.env`. The main directory roots are configured there. Relative paths are resolved from the project root, but absolute paths are also supported.

### Example `.env`

```env
SAVE_DIR=historical_data
RESULTS_DIR=backtest_results
MODEL_DIR=model
```

Example with absolute paths:

```env
SAVE_DIR=C:/Users/your_name/Desktop/EE4016/Project/historical_data
RESULTS_DIR=C:/Users/your_name/Desktop/EE4016/Project/backtest_results
MODEL_DIR=C:/Users/your_name/Desktop/EE4016/Project/model
```

## 3. Run a Syntax Check

Recommended after each code change:

```powershell
uv run python -m py_compile main.py get_stock_data.py train_stock.py predict_stock.py backtest_stock.py run_protocol.py
```

If the command produces no output, the syntax check passed.

## 4. Minimal Working Example: AAPL

### Step 1: Download Data

```powershell
uv run get_stock_data.py --ticker AAPL --years 5 --interval 1d
```

Expected output:

- `historical_data/AAPL_5y_1d.csv`

### Step 2: Train the Model

```powershell
uv run train_stock.py --csv_dir historical_data --ticker AAPL --save_model model.pt --window 30 --epochs 5 --seed 42 --eval_threshold 0.5
```

Expected outputs:

- `model/AAPL/model.pt`
- `model/AAPL/model.scaler.pkl`
- `model/AAPL/model.meta.json`

### Step 3: Run Prediction

```powershell
uv run predict_stock.py --csv historical_data/AAPL_5y_1d.csv --model model/AAPL/model.pt
```

Expected outputs:

- `backtest_results/AAPL/AAPL_5y_1d_pred.csv`
- Possibly `backtest_results/AAPL/AAPL_5y_1d_bad.csv`

### Step 4: Run Backtest

```powershell
uv run backtest_stock.py --csv historical_data/AAPL_5y_1d.csv --model model/AAPL/model.pt --protocol experiment_protocol.json --eval_split test
```

Expected outputs:

- `backtest_results/AAPL/AAPL_5y_1d_bt.csv`
- `backtest_results/AAPL/AAPL_5y_1d_bt_summary.json`

## 5. Protocol Mode

The repository includes `experiment_protocol.json`, which defines:

- A fixed ticker universe
- Fixed data windows
- Fixed split ratios
- Baseline parameters for MACD, RSI, and Bollinger strategies

### 5.1 Download Data Using the Protocol

```powershell
uv run get_stock_data.py --protocol experiment_protocol.json --window_idx 0
```

`window_idx` refers to the `data_windows` index inside `experiment_protocol.json`.

### 5.2 Train with Protocol-Aligned Splits

```powershell
uv run train_stock.py --csv_dir historical_data --ticker AAPL --save_model model.pt --window 30 --epochs 5 --protocol experiment_protocol.json
```

### 5.3 Run the Full Protocol Batch Pipeline

```powershell
uv run run_protocol.py --protocol experiment_protocol.json --window_idxs 0 --epochs 1
```

This automatically performs:

1. Data download
2. Training
3. Backtesting
4. Summary aggregation

Expected output:

- `backtest_results/comparison_summary.csv`

## 6. Design Notes

- Leakage control: the scaler is fit only on the training split, while validation, test, prediction, and backtest data use `transform` only.
- Reproducibility: the pipeline records seeds, parameters, hashes, command metadata, and git information in the metadata file.
- Fair comparison: the model and benchmark strategies are evaluated on the same test segment with the same fee setting.
- Evaluation coverage: both predictive metrics (such as Accuracy and F1) and financial metrics (such as Return, Sharpe, Max Drawdown, and Win Rate) are reported.

## 7. Common Issues and Troubleshooting

### A. `scaler file not found`

Cause: training has not been run yet, or the `--model` path is incorrect.

Fix: verify that `model/<SYMBOL>/*.scaler.pkl` exists and matches the selected model basename.

### B. `Insufficient rows for window`

Cause: the dataset is too short or `window` is too large.

Fix: reduce `--window` or download a longer history, such as `5y`.

### C. `No CSV in ... for ticker ...`

Cause: the expected CSV file is missing from `historical_data`.

Fix: run `get_stock_data.py` first and confirm a filename such as `AAPL_5y_1d.csv` exists.

### D. Highly unstable metrics

Cause: the sample size is too small, the window is too short, or the model was trained for too few epochs.

Fix: keep the seed fixed, increase the data length, increase epochs, and compare methods using the protocol batch workflow instead of a single run.

## Quick Start

The commands below are the fastest pure-CLI path. If you prefer the interactive menu, run `uv run main.py` after `uv sync` and `.env` setup.

```powershell
uv sync
uv run python -m py_compile main.py get_stock_data.py train_stock.py predict_stock.py backtest_stock.py run_protocol.py
uv run get_stock_data.py --protocol experiment_protocol.json --window_idx 0
uv run train_stock.py --csv_dir historical_data --ticker AAPL --save_model model.pt --window 30 --epochs 5 --protocol experiment_protocol.json
uv run predict_stock.py --csv historical_data/AAPL_5y_1d.csv --model model/AAPL/model.pt
uv run backtest_stock.py --csv historical_data/AAPL_5y_1d.csv --model model/AAPL/model.pt --protocol experiment_protocol.json --eval_split test
uv run run_protocol.py --protocol experiment_protocol.json --window_idxs 0 --epochs 1
```

Key outputs:

- Model artifacts: `model/AAPL/model.pt`, `model/AAPL/model.scaler.pkl`, `model/AAPL/model.meta.json`
- Prediction results: `backtest_results/AAPL/AAPL_5y_1d_pred.csv`
- Backtest results: `backtest_results/AAPL/AAPL_5y_1d_bt_summary.json`
- Batch comparison summary: `backtest_results/comparison_summary.csv`
