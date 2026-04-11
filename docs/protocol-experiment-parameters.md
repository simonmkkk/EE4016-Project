# Protocol 實驗參數說明

本文說明 `run_protocol.py` 協定管線與 Train 選單（`train_stock.py`）在參數上的對應與差異。

## 流程對照

| 面向 | Train 選單 | Protocol 管線 (`run_protocol.py`) |
|------|------------|-----------------------------------|
| 資料來源 | 手選一個或多個 `historical_data/*.csv`（須同一 ticker） | 由 protocol JSON 的 `ticker_universe` 與 `data_windows` 驅動；`get_stock_data.py --protocol --window_idx` 寫入 `SAVE_DIR` |
| epochs / window | 選單輸入 | 執行 `run_protocol.py` 時輸入或 CLI（`--epochs`、`--window`） |
| Attention | 可選 Y/n | 管線**未**傳 `--use_attn` 給 `train_stock.py`，預設為**關閉** |
| train/val 比例 | 使用 `train_stock` 預設或自行指定 | 若 JSON 含 `split.method == "ratio"`，會覆寫 `train_ratio` / `val_ratio` |

## Protocol JSON（以 `experiment_protocol.json` 為例）

協定檔定義「實驗設計」層級的參數：

```
----------------------------------------------------------------------
  Protocol Summary（來自 JSON）
----------------------------------------------------------------------
  Protocol name  : default_v2_protocol
  Ticker universe: AAPL, MSFT, NVDA, AMZN, TSLA  （未另指定子集時全跑）
  Data windows   : 2 個
    [0] interval=1d, years=5.0  → 檔名約 {TICKER}_5y_1d.csv
    [1] interval=1h, years=2.0  → 檔名約 {TICKER}_2y_1h.csv
  Split (訓練)   : method=ratio, train_ratio=0.7, val_ratio=0.15
                   （test = 剩餘區段）
  Baselines      : 回測對照策略參數（MACD / RSI / Bollinger 等）
----------------------------------------------------------------------
```

### 欄位摘要

| 欄位 | 用途 |
|------|------|
| `name` | 彙總 CSV 中的 protocol 名稱欄位 |
| `ticker_universe` | 要下載與訓練的股票代號列表 |
| `data_windows[]` | 每個窗的 `interval`（如 `1d`/`1h`）與 `years`（回溯年數） |
| `split` | `train_stock.py` 在 `--protocol` 時讀取的 train/val 比例 |
| `baseline_params` | `backtest_stock.py` 讀取的技術指標與 baseline 設定 |

實際內容以專案根目錄 `experiment_protocol.json` 為準；修改該檔即改變上述摘要。

## 執行 `run_protocol.py` 時的額外參數

這些參數**不在** JSON 內，由選單（`app/actions/stock_actions.py` → `run_protocol_runner`）或命令列指定：

| 參數 | CLI / 互動預設 | 說明 |
|------|----------------|------|
| `epochs` | `5` | 傳給 `train_stock.py --epochs` |
| `window` | `30` | 傳給 `train_stock.py --window` |
| `fee` | `0.001` | 傳給 `backtest_stock.py --fee` |
| `--tickers` | 省略 = 使用 protocol 全部 | 只跑部分 ticker |
| `--window_idxs` | 省略 = 全部窗 | 只跑部分 `data_windows` 索引 |
| `--out` | 預設 `results/comparison_summary.csv` | 策略比較彙總輸出路徑 |

## 相關程式位置

- `run_protocol.py`：下載 → 訓練 → 回測迴圈與彙總
- `experiment_protocol.json`：協定定義
- `train_stock.py`：讀取 `protocol["split"]` 覆寫比例
- `get_stock_data.py`：`--protocol` / `--window_idx`
- `backtest_stock.py`：`--protocol` 載入 `baseline_params`
