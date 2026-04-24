<div align="center">

# 📈 Stock ML — LSTM-Based Stock Direction Prediction & Backtesting System

**EE4016 Applications of AI with Deep Learning — Course Project**

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.10](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

[📖 中文版 README](README_CN.md)

*An end-to-end CLI system for downloading market data, training LSTM direction classifiers across multiple time granularities, backtesting against traditional technical strategies, and generating explainable predictions.*

</div>

---

## Table of Contents

- [Highlights](#highlights)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Configuration](#environment-configuration)
- [Usage](#usage)
  - [Interactive Menu](#interactive-menu)
  - [1. Download Market Data](#1-download-market-data)
  - [2. Train an LSTM Model](#2-train-an-lstm-model)
  - [3. Backtest Against Baselines](#3-backtest-against-baselines)
  - [4. Predict with Explanations](#4-predict-with-explanations)
  - [5. Run Full Pipeline via Protocol](#5-run-full-pipeline-via-protocol)
  - [6. View & Merge Results](#6-view--merge-results)
- [Model Architecture](#model-architecture)
- [Feature Engineering](#feature-engineering)
- [Experiment Protocol](#experiment-protocol)
- [Sample Results](#sample-results)
- [Tech Stack](#tech-stack)

---

## Highlights

| Feature | Description |
|---|---|
| 🧠 **Multi-Granularity LSTM** | Stacked LSTM with residual connections and optional attention, supporting 13 yfinance intervals (1m → 3mo) with learned interval embeddings |
| 📊 **Automated Backtesting** | Side-by-side comparison against Buy & Hold, MACD, RSI, and Bollinger Bands strategies with Sharpe, drawdown, and win-rate metrics |
| 🔍 **Explainable Predictions** | Per-bar explanations citing RSI, MACD, and volume signals; high-confidence error analysis with improvement tips |
| ⚙️ **Protocol-Driven Experiments** | JSON experiment protocol automates the full download → train → backtest pipeline across multiple tickers and time windows |
| 🖥️ **Interactive CLI Menu** | Six sub-menus (Data / Train / Backtest / Predict / Pipeline / Results) with ticker filtering, file pickers, and recap tables |
| 📈 **Adaptive Thresholding** | Validation-set grid search for optimal decision threshold with probability drift adjustment |
| 🔄 **Walk-Forward Evaluation** | Optional rolling-fold evaluation on the test split for robustness checking |

---

## System Architecture

```
┌──────────────────┐     ┌──────────────┐     ┌───────────────────┐     ┌─────────────────────┐
│ Experiment       │────▶│ Protocol     │────▶│ Data Download     │────▶│ historical_data/    │
│ Protocol (.json) │     │ Runner       │     │ (yfinance)        │     │ OHLCV CSVs          │
└──────────────────┘     └──────┬───────┘     └───────────────────┘     └─────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Train LSTM Model    │
                    │ (train_stock.py)    │──────────────────────────────▶ model/<SYMBOL>/
                    └──────────┬──────────┘                               ├── model.pt
                               │                                          ├── model.scaler.pkl
                    ┌──────────▼──────────┐     ┌──────────────────┐      └── model.meta.json
                    │ Backtest & Compare  │────▶│ backtest_results/│
                    │ (backtest_stock.py) │     │ *_bt.csv         │
                    └──────────┬──────────┘     │ *_bt_summary.json│
                               │                └──────────────────┘
                    ┌──────────▼──────────┐
                    │ Predict & Explain   │──────────────────────────────▶ *_pred.csv
                    │ (predict_stock.py)  │                               *_bad.csv (errors)
                    └─────────────────────┘
```

**Strategies compared during backtesting:**
- **Model (LSTM)** — probability → threshold → long / short signal
- **Buy & Hold** — always long
- **MACD** — MACD line vs signal line crossover
- **RSI** — oversold / overbought bands
- **Bollinger Bands** — price vs upper / lower band breakout

---

## Project Structure

```
EE4016-Project/
├── main.py                        # Entry point — launches interactive CLI menu
├── pyproject.toml                 # Project metadata & dependencies (uv / pip)
├── experiment_protocol.json       # Default experiment protocol
├── .env_example                   # Environment variable template
│
├── app/                           # Application package
│   ├── menu.py                    #   Main menu dispatcher
│   ├── model.py                   #   LSTMDir model definition
│   ├── fe.py                      #   Feature engineering (technical indicators)
│   ├── sequences.py               #   Windowed tensor builder & evaluation
│   ├── constants.py               #   Interval definitions & mappings
│   ├── paths.py                   #   .env-driven path resolution
│   ├── state.py                   #   Persistent menu state (JSON)
│   │
│   ├── workflows/                 #   Core workflow scripts
│   │   ├── get_stock_data.py      #     Download OHLCV via yfinance
│   │   ├── train_stock.py         #     Train LSTM direction classifier
│   │   ├── backtest_stock.py      #     Backtest model vs baselines
│   │   ├── predict_stock.py       #     Inference with explainability
│   │   ├── run_protocol.py        #     Full pipeline orchestrator
│   │   └── merge_all_results.py   #     Merge all results into one CSV
│   │
│   ├── actions/                   #   Menu action handlers
│   ├── menus/                     #   Sub-menu definitions
│   ├── services/                  #   File discovery & picker utilities
│   ├── domain/                    #   Dataset loading & frame builders
│   ├── infra/                     #   Artifact loading, JSON/Pickle I/O
│   ├── reporting/                 #   Results flattening & merging
│   └── ui/                        #   CLI helpers (clear screen, prompts)
│
├── historical_data/               # Downloaded OHLCV CSVs (git-ignored)
├── model/                         # Trained models per symbol (git-ignored)
│   └── <SYMBOL>/
│       ├── model.pt
│       ├── model.scaler.pkl
│       └── model.meta.json
├── backtest_results/              # Backtest & predict outputs (git-ignored)
│   └── <SYMBOL>/
│       ├── *_bt.csv
│       ├── *_bt_summary.json
│       └── *_pred.csv
├── flowchart/                     # System architecture diagrams
└── result photo/                  # Sample result screenshots
```

---

## Getting Started

### Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.12 |
| CUDA (optional, for GPU training) | ≥ 13.0 |
| [uv](https://docs.astral.sh/uv/) (recommended) | latest |

### Installation

**Option A — Using `uv` (recommended):**

```bash
# Clone the repository
git clone https://github.com/codylam1228/EE4016-Project.git
cd EE4016-Project

# Create virtual environment and install all dependencies
uv sync
```

**Option B — Using `pip`:**

```bash
git clone https://github.com/codylam1228/EE4016-Project.git
cd EE4016-Project

python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e .
```

> **Note:** PyTorch is pulled from the CUDA 13.0 wheel index by default. If you don't have an NVIDIA GPU, modify `pyproject.toml` to use the CPU-only index.

### Environment Configuration

```bash
# Copy the template and edit as needed
cp .env_example .env
```

| Variable | Default | Description |
|---|---|---|
| `SAVE_DIR` | `historical_data` | Directory for downloaded OHLCV CSVs |
| `RESULTS_DIR` | `backtest_results` | Directory for backtest / predict outputs |
| `MODEL_DIR` | `model` | Directory for trained model artifacts |

All paths are relative to the project root unless specified as absolute paths.

---

## Usage

### Interactive Menu

Launch the system with:

```bash
python main.py
```

This opens the main menu with six sub-menus:

```
======================================================================
Stock ML - Training & Backtesting System
======================================================================

Please select an option:

[Data]
   1. Data menu

[Train]
   2. Train menu

[Backtest]
   3. Backtest menu

[Predict]
   4. Predict menu

[Pipeline]
   5. Pipeline menu

[Results]
   6. Results menu

   0. Exit
======================================================================
```

Each workflow can also be invoked programmatically with CLI arguments (see below).

---

### 1. Download Market Data

Downloads OHLCV data from Yahoo Finance for one or more tickers at specified intervals.

**Via menu:** Select `1. Data menu` → choose tickers, intervals, and lookback period.

**Via CLI:**

```bash
# Single ticker + interval
python -m app.workflows.get_stock_data \
    --ticker AAPL --interval 1d 1h --years 5

# Via experiment protocol
python -m app.workflows.get_stock_data \
    --protocol experiment_protocol.json --window_idx 0
```

**Supported intervals:** `1m`, `2m`, `5m`, `15m`, `30m`, `60m`, `90m`, `1h`, `1d`, `5d`, `1wk`, `1mo`, `3mo`

Output: `historical_data/{TICKER}_{interval}_{lookback}.csv`

---

### 2. Train an LSTM Model

Trains a multi-granularity LSTM direction classifier with extensive hyperparameter control.

**Via menu:** Select `2. Train menu` → pick CSVs, set epochs, window, attention, label mode.

**Via CLI:**

```bash
python -m app.workflows.train_stock \
    --csv_dir historical_data --ticker AAPL \
    --save_model model.pt \
    --window 100 --epochs 100 \
    --use_attn \
    --label_threshold_quantile 0.55
```

**Key training arguments:**

| Argument | Default | Description |
|---|---|---|
| `--window` | `30` | Lookback window length (number of bars) |
| `--epochs` | `40` | Maximum training epochs |
| `--batch` | `256` | Mini-batch size |
| `--lr` | `1e-3` | Initial learning rate |
| `--patience` | `15` | Early stopping patience |
| `--use_attn` | `false` | Enable attention pooling |
| `--num_layers` | `2` | Number of stacked LSTM layers |
| `--dropout` | `0.4` | Dropout before FC layer |
| `--train_ratio` | `0.7` | Train split proportion |
| `--val_ratio` | `0.15` | Validation split proportion |
| `--threshold_objective` | `f1_macro` | Objective for threshold search (`f1_pos`, `f1_macro`, `balanced_accuracy`, `mcc`) |
| `--label_threshold_quantile` | `None` | Per-series adaptive labelling (e.g., `0.55` = top 45% as UP) |

**Output artifacts** (saved to `model/<SYMBOL>/`):
- `model.pt` — trained model state dict
- `model.scaler.pkl` — fitted StandardScaler
- `model.meta.json` — full training metadata (features, hyperparameters, metrics, reproducibility info)

---

### 3. Backtest Against Baselines

Evaluates the trained LSTM model against four traditional strategies on historical data.

**Via menu:** Select `3. Backtest menu` → pick CSV(s), model, threshold, fee.

**Via CLI:**

```bash
python -m app.workflows.backtest_stock \
    --csv historical_data/AAPL_1d_10y.csv \
    --model model/AAPL/model.pt \
    --fee 0.001 --eval_split test
```

**Metrics reported per strategy:**
- Total Return, Sharpe Ratio, Max Drawdown, Win Rate, Turnover Rate
- Direction Accuracy & F1 Score (for the LSTM model)

**Output:** `backtest_results/<SYMBOL>/<stem>_bt.csv` + `<stem>_bt_summary.json`

---

### 4. Predict with Explanations

Generates per-bar predictions with human-readable explanations and error analysis.

**Via menu:** Select `4. Predict menu` → pick CSV(s) and model.

**Via CLI:**

```bash
python -m app.workflows.predict_stock \
    --csv historical_data/AAPL_1d_10y.csv \
    --model model/AAPL/model.pt
```

**Each row contains:**
- `pred_prob` — model probability of UP
- `prediction` — binary direction (UP / DOWN)
- `explanation` — natural language reason (e.g., *"up bias: RSI<30 (oversold); MACD positive"*)
- `why_wrong` — explanation of mispredictions
- `improve_tip` — suggestions for improving accuracy
- `high_conf_wrong` — flagged high-confidence errors

**Output:** `backtest_results/<SYMBOL>/<stem>_pred.csv` (+ `_bad.csv` for high-confidence errors)

---

### 5. Run Full Pipeline via Protocol

Automates the entire workflow (download → train → backtest) across multiple tickers and data windows using a JSON protocol file.

**Via menu:** Select `5. Pipeline menu` → pick protocol, set epochs/window/fee.

**Via CLI:**

```bash
python -m app.workflows.run_protocol \
    --protocol experiment_protocol.json \
    --epochs 100 --window 100 --fee 0.001
```

**Output:** `backtest_results/comparison_summary.csv` — aggregated strategy comparison across all ticker × window combinations.

---

### 6. View & Merge Results

**Via menu:** Select `6. Results menu` to:
- View merged results across all backtested CSVs
- Auto-generate missing prediction files
- Export a unified `all_results_merged.csv`

---

## Model Architecture

The core model is `LSTMDir` — a stacked LSTM binary classifier for predicting the next-bar price direction.

```
Input (B, T, d_in)
       │
       ▼
┌──────────────┐
│ LSTM Layer 1 │
└──────┬───────┘
       │
┌──────▼───────┐
│ LSTM Layer 2 │◄── Residual connection (output + input)
└──────┬───────┘
       │
       ├─── (if attention) ──▶ Softmax-weighted sum over T
       │
       └─── (default) ──────▶ Last hidden state
       │
       ▼
   Dropout(p)
       │
       ├─── (if interval embedding) ──▶ Concat with Embedding(interval_id)
       │
       ▼
   Linear → logit (BCEWithLogitsLoss)
```

**Key design decisions:**
- **Residual connections** between LSTM layers (layer 2+ adds its input to its output) to ease gradient flow
- **Optional attention** (`--use_attn`) replaces last-hidden-state pooling with learned attention weights
- **Interval embeddings** (8-dim) allow a single model to learn interval-specific patterns when trained on mixed-granularity data
- **Balanced pos_weight** in BCEWithLogitsLoss compensates for class imbalance
- **ReduceLROnPlateau** scheduler halves the learning rate after a configurable patience window

---

## Feature Engineering

The feature engineering pipeline (`app/fe.py`) computes the following technical indicators from raw OHLCV data:

| Feature | Description | Library |
|---|---|---|
| `rsi` | Relative Strength Index (window=14) | `ta.momentum` |
| `macd` | MACD histogram (12/26/9) | `ta.trend` |
| `bbw` | Bollinger Band Width (window=20) | `ta.volatility` |
| `atr` | Average True Range (window=14) | `ta.volatility` |
| `vma20` | Volume SMA (window=20) | `ta.trend` |
| `v_ratio` | Volume / VMA20 | computed |
| `body` | |Open − Close| | computed |
| `range` | High − Low | computed |
| `granularity` | 0 = daily, 1 = intraday | computed from timestamp |
| `interval_id` | Integer encoding of yfinance interval | computed from filename |

All indicator windows are clamped to the series length so that short CSVs (e.g., 1m/6d) don't crash.

---

## Experiment Protocol

The experiment protocol (`experiment_protocol.json`) defines reproducible experiment configurations:

```json
{
  "name": "default_v2_protocol",
  "ticker_universe": ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA"],
  "data_windows": [
    { "interval": "1d", "years": 5.0 },
    { "interval": "1h", "years": 2.0 }
  ],
  "split": {
    "method": "ratio",
    "train_ratio": 0.7,
    "val_ratio": 0.15
  },
  "baseline_params": {
    "buy_and_hold": {},
    "macd": { "fast": 12, "slow": 26, "signal": 9 },
    "rsi": { "window": 14, "oversold": 30, "overbought": 70 },
    "bollinger": { "window": 20, "std": 2.0 }
  }
}
```

The protocol runner (`run_protocol.py`) iterates over every `ticker × data_window` combination to produce a unified `comparison_summary.csv`.

---

## Sample Results

### Training Metrics (AAPL)

| Split | N | Accuracy | Precision | Recall | F1 | Threshold |
|---|---|---|---|---|---|---|
| VAL | 2,473 | 0.5317 | 0.4846 | 0.3613 | 0.4140 | 0.50 |
| TEST | 2,475 | 0.5063 | 0.4484 | 0.5266 | 0.4844 | 0.50 |

### Strategy Comparison — Total Return (AAPL, multi-granularity)

| CSV | LSTM | Buy&Hold | MACD | RSI | Bollinger |
|---|---|---|---|---|---|
| AAPL_5d_10y | **+1.0023** | +0.1014 | +0.2092 | +0.1845 | +0.1098 |
| AAPL_1wk_10y | **+0.5773** | +0.0643 | +0.3999 | +0.0000 | +0.0848 |
| AAPL_1d_10y | +0.0269 | **+0.1798** | −0.2560 | +0.0389 | −0.2153 |
| AAPL_15m_59d | **+0.0684** | +0.0195 | +0.0435 | −0.0180 | −0.0434 |
| AAPL_1h_2y | **+0.0875** | −0.0441 | +0.0110 | −0.0567 | −0.0875 |

> The LSTM model outperforms all baselines on most timeframes, with particularly strong results on lower-frequency data (5d, 1wk).

---

## Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python 3.12+ |
| **Deep Learning** | PyTorch 2.10 (CUDA 13.0) |
| **Data** | pandas, NumPy, yfinance |
| **Technical Analysis** | [ta](https://github.com/bukosabino/ta) (RSI, MACD, Bollinger, ATR) |
| **Preprocessing** | scikit-learn (StandardScaler) |
| **Model Inspection** | torchinfo |
| **Visualization** | matplotlib |
| **Package Management** | uv / pip |
| **Environment** | python-dotenv |


