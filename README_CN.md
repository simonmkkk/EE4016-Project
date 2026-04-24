<div align="center">

# 📈 Stock ML — 基於 LSTM 的股票方向預測與回測系統

**EE4016 人工智能與深度學習應用 — 課程專題**

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.10](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

[📖 English README](README.md)

*一個端到端的 CLI 系統，支援下載市場數據、跨多種時間粒度訓練 LSTM 方向分類器、與傳統技術策略進行回測對比，並生成可解釋的預測結果。*

</div>

---

## 目錄

- [項目亮點](#項目亮點)
- [系統架構](#系統架構)
- [項目結構](#項目結構)
- [快速開始](#快速開始)
  - [環境要求](#環境要求)
  - [安裝步驟](#安裝步驟)
  - [環境配置](#環境配置)
- [使用方法](#使用方法)
  - [互動式選單](#互動式選單)
  - [1. 下載市場數據](#1-下載市場數據)
  - [2. 訓練 LSTM 模型](#2-訓練-lstm-模型)
  - [3. 回測與基準比較](#3-回測與基準比較)
  - [4. 預測與解釋](#4-預測與解釋)
  - [5. 通過協議運行完整管道](#5-通過協議運行完整管道)
  - [6. 查看與合併結果](#6-查看與合併結果)
- [模型架構](#模型架構)
- [特徵工程](#特徵工程)
- [實驗協議](#實驗協議)
- [示例結果](#示例結果)
- [技術棧](#技術棧)

---

## 項目亮點

| 功能 | 描述 |
|---|---|
| 🧠 **多粒度 LSTM** | 堆疊式 LSTM，支援殘差連接和可選注意力機制，覆蓋 13 種 yfinance 時間間隔（1m → 3mo），配備學習型時間間隔嵌入 |
| 📊 **自動化回測** | 與 Buy & Hold、MACD、RSI、Bollinger Bands 四種傳統策略進行並排比較，報告夏普比率、最大回撤、勝率等指標 |
| 🔍 **可解釋預測** | 逐條生成自然語言解釋（引用 RSI、MACD、成交量信號）；高置信度錯誤分析與改進建議 |
| ⚙️ **協議驅動實驗** | JSON 實驗協議自動化執行「下載 → 訓練 → 回測」全流程，支援多個股票和時間窗口 |
| 🖥️ **互動式 CLI 選單** | 六大子選單（數據 / 訓練 / 回測 / 預測 / 管道 / 結果），支援股票篩選、檔案選擇器、結果彙總表 |
| 📈 **自適應閾值** | 驗證集網格搜索最優決策閾值，支援概率漂移調整 |
| 🔄 **滾動前推驗證** | 可選的測試集滾動折疊評估，用於穩健性檢驗 |

---

## 系統架構

```
┌──────────────────┐     ┌──────────────┐     ┌───────────────────┐     ┌─────────────────────┐
│ 實驗協議         │────▶│ 協議運行器   │────▶│ 數據下載          │────▶│ historical_data/    │
│ (.json)          │     │ run_protocol │     │ (yfinance)        │     │ OHLCV CSV 原始數據  │
└──────────────────┘     └──────┬───────┘     └───────────────────┘     └─────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │ 訓練 LSTM 模型      │
                    │ (train_stock.py)    │──────────────────────────────▶ model/<SYMBOL>/
                    └──────────┬──────────┘                               ├── model.pt
                               │                                          ├── model.scaler.pkl
                    ┌──────────▼──────────┐     ┌──────────────────┐      └── model.meta.json
                    │ 回測與基準比較      │────▶│ backtest_results/│
                    │ (backtest_stock.py) │     │ *_bt.csv         │
                    └──────────┬──────────┘     │ *_bt_summary.json│
                               │                └──────────────────┘
                    ┌──────────▼──────────┐
                    │ 預測與解釋          │──────────────────────────────▶ *_pred.csv
                    │ (predict_stock.py)  │                               *_bad.csv (錯誤分析)
                    └─────────────────────┘
```

**回測比較的策略：**
- **模型策略 (LSTM)** — 概率 → 閾值 → 做多 / 做空信號
- **買入持有 (Buy & Hold)** — 始終做多
- **MACD** — MACD 線與信號線交叉
- **RSI** — 超賣 / 超買區間
- **布林帶 (Bollinger Bands)** — 價格突破上軌 / 下軌

---

## 項目結構

```
EE4016-Project/
├── main.py                        # 入口 — 啟動互動式 CLI 選單
├── pyproject.toml                 # 項目元數據和依賴（uv / pip）
├── experiment_protocol.json       # 默認實驗協議
├── .env_example                   # 環境變量模板
│
├── app/                           # 應用包
│   ├── menu.py                    #   主選單分發器
│   ├── model.py                   #   LSTMDir 模型定義
│   ├── fe.py                      #   特徵工程（技術指標）
│   ├── sequences.py               #   滑動窗口張量構建與評估
│   ├── constants.py               #   時間間隔定義與映射
│   ├── paths.py                   #   基於 .env 的路徑解析
│   ├── state.py                   #   選單狀態持久化（JSON）
│   │
│   ├── workflows/                 #   核心工作流腳本
│   │   ├── get_stock_data.py      #     通過 yfinance 下載 OHLCV
│   │   ├── train_stock.py         #     訓練 LSTM 方向分類器
│   │   ├── backtest_stock.py      #     回測模型 vs 基準策略
│   │   ├── predict_stock.py       #     推論與可解釋性
│   │   ├── run_protocol.py        #     完整管道編排器
│   │   └── merge_all_results.py   #     合併所有結果為一個 CSV
│   │
│   ├── actions/                   #   選單操作處理器
│   ├── menus/                     #   子選單定義
│   ├── services/                  #   檔案發現與選擇器工具
│   ├── domain/                    #   數據集載入與資料框構建
│   ├── infra/                     #   模型加載、JSON/Pickle I/O
│   ├── reporting/                 #   結果扁平化與合併
│   └── ui/                        #   CLI 輔助工具（清屏、提示）
│
├── historical_data/               # 下載的 OHLCV CSV（git 忽略）
├── model/                         # 按股票代碼分目錄保存的模型（git 忽略）
│   └── <SYMBOL>/
│       ├── model.pt
│       ├── model.scaler.pkl
│       └── model.meta.json
├── backtest_results/              # 回測和預測輸出（git 忽略）
│   └── <SYMBOL>/
│       ├── *_bt.csv
│       ├── *_bt_summary.json
│       └── *_pred.csv
├── flowchart/                     # 系統架構圖
└── result photo/                  # 示例結果截圖
```

---

## 快速開始

### 環境要求

| 條件 | 版本 |
|---|---|
| Python | ≥ 3.12 |
| CUDA（可選，用於 GPU 訓練） | ≥ 13.0 |
| [uv](https://docs.astral.sh/uv/)（推薦） | 最新版 |

### 安裝步驟

**方式 A — 使用 `uv`（推薦）：**

```bash
# 克隆倉庫
git clone https://github.com/codylam1228/EE4016-Project.git
cd EE4016-Project

# 創建虛擬環境並安裝所有依賴
uv sync
```

**方式 B — 使用 `pip`：**

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

> **注意：** 默認從 CUDA 13.0 wheel 索引拉取 PyTorch。如果沒有 NVIDIA GPU，請修改 `pyproject.toml` 使用 CPU 版本索引。

### 環境配置

```bash
# 複製模板並按需修改
cp .env_example .env
```

| 變量 | 默認值 | 說明 |
|---|---|---|
| `SAVE_DIR` | `historical_data` | 下載的 OHLCV CSV 存放目錄 |
| `RESULTS_DIR` | `backtest_results` | 回測 / 預測輸出目錄 |
| `MODEL_DIR` | `model` | 訓練好的模型存放目錄 |

所有路徑相對於項目根目錄，除非指定為絕對路徑。

---

## 使用方法

### 互動式選單

啟動系統：

```bash
python main.py
```

將顯示主選單，包含六個子選單：

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

每個工作流也可以通過 CLI 參數直接調用（見下文）。

---

### 1. 下載市場數據

從 Yahoo Finance 下載一個或多個股票的 OHLCV 數據。

**通過選單：** 選擇 `1. Data menu` → 輸入股票代碼、時間間隔和回溯期。

**通過 CLI：**

```bash
# 單個股票 + 多個間隔
python -m app.workflows.get_stock_data \
    --ticker AAPL --interval 1d 1h --years 5

# 通過實驗協議
python -m app.workflows.get_stock_data \
    --protocol experiment_protocol.json --window_idx 0
```

**支援的時間間隔：** `1m`、`2m`、`5m`、`15m`、`30m`、`60m`、`90m`、`1h`、`1d`、`5d`、`1wk`、`1mo`、`3mo`

輸出：`historical_data/{TICKER}_{interval}_{lookback}.csv`

---

### 2. 訓練 LSTM 模型

訓練多粒度 LSTM 方向分類器，支援豐富的超參數控制。

**通過選單：** 選擇 `2. Train menu` → 選擇 CSV、設定 epochs、window、注意力、標籤模式。

**通過 CLI：**

```bash
python -m app.workflows.train_stock \
    --csv_dir historical_data --ticker AAPL \
    --save_model model.pt \
    --window 100 --epochs 100 \
    --use_attn \
    --label_threshold_quantile 0.55
```

**主要訓練參數：**

| 參數 | 默認值 | 說明 |
|---|---|---|
| `--window` | `30` | 回溯窗口長度（K 線數量） |
| `--epochs` | `40` | 最大訓練輪數 |
| `--batch` | `256` | 小批量大小 |
| `--lr` | `1e-3` | 初始學習率 |
| `--patience` | `15` | 早停耐心值 |
| `--use_attn` | `false` | 啟用注意力池化 |
| `--num_layers` | `2` | 堆疊 LSTM 層數 |
| `--dropout` | `0.4` | FC 層前的 Dropout |
| `--train_ratio` | `0.7` | 訓練集比例 |
| `--val_ratio` | `0.15` | 驗證集比例 |
| `--threshold_objective` | `f1_macro` | 閾值搜索目標（`f1_pos`、`f1_macro`、`balanced_accuracy`、`mcc`） |
| `--label_threshold_quantile` | `None` | 逐序列自適應標籤（例如 `0.55` = 前 45% 標為上漲） |

**輸出文件**（保存至 `model/<SYMBOL>/`）：
- `model.pt` — 訓練好的模型權重
- `model.scaler.pkl` — 擬合的 StandardScaler
- `model.meta.json` — 完整訓練元數據（特徵、超參數、指標、可重現性信息）

---

### 3. 回測與基準比較

將訓練好的 LSTM 模型與四種傳統策略在歷史數據上進行對比評估。

**通過選單：** 選擇 `3. Backtest menu` → 選擇 CSV、模型、閾值、手續費。

**通過 CLI：**

```bash
python -m app.workflows.backtest_stock \
    --csv historical_data/AAPL_1d_10y.csv \
    --model model/AAPL/model.pt \
    --fee 0.001 --eval_split test
```

**每種策略報告的指標：**
- 總收益、夏普比率、最大回撤、勝率、換手率
- 方向準確率與 F1 分數（僅 LSTM 模型）

**輸出：** `backtest_results/<SYMBOL>/<stem>_bt.csv` + `<stem>_bt_summary.json`

---

### 4. 預測與解釋

生成逐條帶有自然語言解釋的預測結果和錯誤分析。

**通過選單：** 選擇 `4. Predict menu` → 選擇 CSV 和模型。

**通過 CLI：**

```bash
python -m app.workflows.predict_stock \
    --csv historical_data/AAPL_1d_10y.csv \
    --model model/AAPL/model.pt
```

**每行包含：**
- `pred_prob` — 模型預測的上漲概率
- `prediction` — 二元方向（上漲 / 下跌）
- `explanation` — 自然語言解釋（例如：*"up bias: RSI<30 (oversold); MACD positive"*）
- `why_wrong` — 錯誤預測的原因分析
- `improve_tip` — 改進準確率的建議
- `high_conf_wrong` — 標記的高置信度錯誤

**輸出：** `backtest_results/<SYMBOL>/<stem>_pred.csv`（+ `_bad.csv` 高置信度錯誤）

---

### 5. 通過協議運行完整管道

使用 JSON 協議文件自動化執行整個工作流（下載 → 訓練 → 回測），涵蓋多個股票和數據窗口。

**通過選單：** 選擇 `5. Pipeline menu` → 選擇協議、設定 epochs/window/fee。

**通過 CLI：**

```bash
python -m app.workflows.run_protocol \
    --protocol experiment_protocol.json \
    --epochs 100 --window 100 --fee 0.001
```

**輸出：** `backtest_results/comparison_summary.csv` — 所有股票 × 窗口組合的策略比較彙總。

---

### 6. 查看與合併結果

**通過選單：** 選擇 `6. Results menu` 可以：
- 查看所有已回測 CSV 的合併結果
- 自動生成缺失的預測文件
- 導出統一的 `all_results_merged.csv`

---

## 模型架構

核心模型為 `LSTMDir` — 一個堆疊式 LSTM 二元分類器，用於預測下一根 K 線的價格方向。

```
輸入 (B, T, d_in)
       │
       ▼
┌──────────────┐
│ LSTM 第 1 層 │
└──────┬───────┘
       │
┌──────▼───────┐
│ LSTM 第 2 層 │◄── 殘差連接（輸出 + 輸入）
└──────┬───────┘
       │
       ├─── (如有注意力) ──▶ Softmax 加權求和（沿 T 維度）
       │
       └─── (默認) ────────▶ 最後一個隱藏狀態
       │
       ▼
   Dropout(p)
       │
       ├─── (如有間隔嵌入) ──▶ 與 Embedding(interval_id) 拼接
       │
       ▼
   Linear → logit (BCEWithLogitsLoss)
```

**關鍵設計決策：**
- **殘差連接** — 第 2 層以上的 LSTM 輸出與輸入相加，有助於梯度流動
- **可選注意力** (`--use_attn`) — 用學習的注意力權重替代取最後隱藏狀態
- **時間間隔嵌入**（8 維）— 允許單一模型在混合粒度數據上學習間隔特定的模式
- **平衡 pos_weight** — BCEWithLogitsLoss 中自動計算正負樣本權重以補償類別不平衡
- **ReduceLROnPlateau 調度器** — 在可配置的耐心窗口後將學習率減半

---

## 特徵工程

特徵工程流水線（`app/fe.py`）從原始 OHLCV 數據計算以下技術指標：

| 特徵 | 描述 | 來源 |
|---|---|---|
| `rsi` | 相對強弱指數（窗口=14） | `ta.momentum` |
| `macd` | MACD 柱狀圖（12/26/9） | `ta.trend` |
| `bbw` | 布林帶寬度（窗口=20） | `ta.volatility` |
| `atr` | 平均真實波幅（窗口=14） | `ta.volatility` |
| `vma20` | 成交量 SMA（窗口=20） | `ta.trend` |
| `v_ratio` | 成交量 / VMA20 | 計算得出 |
| `body` | |開盤價 − 收盤價| | 計算得出 |
| `range` | 最高價 − 最低價 | 計算得出 |
| `granularity` | 0 = 日線，1 = 日內 | 由時間戳推斷 |
| `interval_id` | yfinance 時間間隔的整數編碼 | 由檔案名推斷 |

所有指標的窗口長度會被 clamp 到序列長度，確保短 CSV（如 1m/6d）不會崩潰。

---

## 實驗協議

實驗協議（`experiment_protocol.json`）定義可重現的實驗配置：

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

協議運行器（`run_protocol.py`）會遍歷每一個 `股票 × 數據窗口` 組合，生成統一的 `comparison_summary.csv`。

---

## 示例結果

### 訓練指標（AAPL）

| 集合 | 樣本數 | 準確率 | 精確率 | 召回率 | F1 | 閾值 |
|---|---|---|---|---|---|---|
| 驗證集 | 2,473 | 0.5317 | 0.4846 | 0.3613 | 0.4140 | 0.50 |
| 測試集 | 2,475 | 0.5063 | 0.4484 | 0.5266 | 0.4844 | 0.50 |

### 策略比較 — 總收益（AAPL，多粒度）

| CSV | LSTM | 買入持有 | MACD | RSI | 布林帶 |
|---|---|---|---|---|---|
| AAPL_5d_10y | **+1.0023** | +0.1014 | +0.2092 | +0.1845 | +0.1098 |
| AAPL_1wk_10y | **+0.5773** | +0.0643 | +0.3999 | +0.0000 | +0.0848 |
| AAPL_1d_10y | +0.0269 | **+0.1798** | −0.2560 | +0.0389 | −0.2153 |
| AAPL_15m_59d | **+0.0684** | +0.0195 | +0.0435 | −0.0180 | −0.0434 |
| AAPL_1h_2y | **+0.0875** | −0.0441 | +0.0110 | −0.0567 | −0.0875 |

> LSTM 模型在大多數時間框架上優於所有基準策略，在低頻數據（5d、1wk）上表現尤為突出。

---

## 技術棧

| 類別 | 技術 |
|---|---|
| **語言** | Python 3.12+ |
| **深度學習** | PyTorch 2.10（CUDA 13.0） |
| **數據處理** | pandas、NumPy、yfinance |
| **技術分析** | [ta](https://github.com/bukosabino/ta)（RSI、MACD、布林帶、ATR） |
| **預處理** | scikit-learn（StandardScaler） |
| **模型檢查** | torchinfo |
| **可視化** | matplotlib |
| **包管理** | uv / pip |
| **環境管理** | python-dotenv |


