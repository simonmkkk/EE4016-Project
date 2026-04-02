# Adaptive Multi-Granularity Neural Trading System

這份文件會一步一步教你如何在本專案中執行：

1. 資料下載 (`get_stock_data.py`)  
2. 模型訓練 (`train_stock.py`)  
3. 推論 (`predict_stock.py`)  
4. 回測 (`backtest_stock.py`)  
5. Protocol 批次實驗 (`run_protocol.py`)

## Flow Diagram

![Project Flow Diagram](flowchart/flowchart.png)

流程圖展示了本專案的端到端管線：`experiment_protocol.json` 由 `run_protocol.py` 協調後，驅動資料下載、模型訓練與回測比較。訓練會產生 `model/<SYMBOL>/` 下的模型與可重現 artifacts（`pt/scaler/meta`），推論可獨立執行並輸出解釋欄位，而回測會在相同評估區段下比較 LSTM 與基準策略，最終輸出到 `RESULTS_DIR`（預設 `backtest_results/`，於 `.env` 設定；含明細、summary 與 comparison 總表）。

**互動式入口（新）**：在專案根目錄執行 `uv run main.py` 可開啟選單（`app/` 內 Data / Train / Backtest / Predict / Pipeline / Results）。選單會從 `historical_data/*.csv` 挑檔；請將 `.env` 的 `SAVE_DIR` 指到同一資料夾（與 `get_stock_data` 預設一致）。細節見 `MAIN_TO_FULL_PROJECT_FLOW.md`。

### Project structure

```text
EE4016-Project
|
+-- main.py
+-- get_stock_data.py
+-- train_stock.py
+-- backtest_stock.py
+-- predict_stock.py
+-- run_protocol.py
|
+-- app
|   |
|   +-- menu.py
|   +-- state.py
|   +-- constants.py
|   +-- paths.py
|   |
|   +-- actions
|   |   `-- stock_actions.py
|   |
|   +-- services
|   |   `-- discovery.py
|   |
|   +-- ui
|   |   `-- cli.py
|   |
|   `-- menus
|       +-- data_menu.py
|       +-- train_menu.py
|       +-- backtest_menu.py
|       +-- pipeline_menu.py
|       +-- predict_menu.py
|       `-- results_menu.py
|
+-- historical_data/
+-- model/
+-- backtest_results/
|
+-- MAIN_TO_DATA_DOWNLOAD_FLOW.md
+-- MAIN_TO_DATA_DOWNLOAD_CODE_PATH.md
`-- MAIN_TO_FULL_PROJECT_FLOW.md
```

---

## 1) 環境需求

- Windows / macOS / Linux
- Python 3.12+
- `uv` 套件管理工具

若尚未安裝 `uv`，可參考官方文件安裝。安裝後，在專案根目錄執行：

```powershell
uv sync
```

這會建立 `.venv` 並安裝相依套件。

---

## 2) 先設定 `.env`（第一次執行必做）

複製 `.env_example` 為 `.env`。所有主要資料夾路徑都在此設定（相對路徑以專案根目錄為準，亦可用絕對路徑）。

### `.env` 範例

```env
SAVE_DIR=historical_data
RESULTS_DIR=backtest_results
MODEL_DIR=model
```

或使用絕對路徑，例如：

```env
SAVE_DIR=C:/Users/your_name/Desktop/EE4016/Project/historical_data
RESULTS_DIR=C:/Users/your_name/Desktop/EE4016/Project/backtest_results
MODEL_DIR=C:/Users/your_name/Desktop/EE4016/Project/model
```

設定完成後，再進行語法檢查。

## 3) 先做語法檢查（建議每次改完先跑）

```powershell
uv run python -m py_compile main.py get_stock_data.py train_stock.py predict_stock.py backtest_stock.py run_protocol.py
```

若沒有任何輸出，代表語法檢查通過。

---

## 4) 最小可跑流程（單一股票 AAPL）

### Step 1: 下載資料

```powershell
uv run get_stock_data.py --ticker AAPL --years 5 --interval 1d
```

預期輸出檔案：

- `historical_data/AAPL_5y_1d.csv`

### Step 2: 訓練模型

```powershell
uv run train_stock.py --csv_dir historical_data --ticker AAPL --save_model dir_model.pt --window 30 --epochs 5 --seed 42 --eval_threshold 0.5
```

預期輸出檔案：

- `model/AAPL/dir_model.pt`
- `model/AAPL/dir_model.scaler.pkl`
- `model/AAPL/dir_model.meta.json`

### Step 3: 推論

```powershell
uv run predict_stock.py --csv historical_data/AAPL_5y_1d.csv --model model/AAPL/dir_model.pt
```

預期輸出檔案：

- `backtest_results/AAPL/AAPL_5y_1d_pred.csv`
- （可能）`backtest_results/AAPL/AAPL_5y_1d_bad.csv`

### Step 4: 回測（含 baseline 比較）

```powershell
uv run backtest_stock.py --csv historical_data/AAPL_5y_1d.csv --model model/AAPL/dir_model.pt --protocol experiment_protocol.json --eval_split test
```

預期輸出檔案：

- `backtest_results/AAPL/AAPL_5y_1d_bt.csv`
- `backtest_results/AAPL/AAPL_5y_1d_bt_summary.json`

---

## 5) Protocol 模式（建議報告展示用）

專案已提供 `experiment_protocol.json`，內含：

- 固定 ticker universe
- 固定 data windows
- 固定 split ratio
- baseline 參數（MACD / RSI / Bollinger）

### 4.1 用 protocol 下載資料

```powershell
uv run get_stock_data.py --protocol experiment_protocol.json --window_idx 0
```

`window_idx` 對應 `experiment_protocol.json` 的 `data_windows` 索引。

### 4.2 用 protocol 訓練（split 自動對齊）

```powershell
uv run train_stock.py --csv_dir historical_data --ticker AAPL --save_model dir_model.pt --window 30 --epochs 5 --protocol experiment_protocol.json
```

### 4.3 一鍵跑 protocol 批次流程

```powershell
uv run run_protocol.py --protocol experiment_protocol.json --window_idxs 0 --epochs 1
```

這會自動執行：

1. 下載資料  
2. 訓練  
3. 回測  
4. 輸出彙總比較表

預期輸出檔案：

- `backtest_results/comparison_summary.csv`

---

## 6) 重要設計說明（你在報告可引用）

- **Leakage control**：Scaler 僅在 train split `fit`，val/test/predict/backtest 只 `transform`。  
- **Reproducibility**：固定 seed + DataLoader generator；metadata 記錄參數、版本、SHA256、命令、git hash。  
- **Fair comparison**：Model 與 baseline 在相同 test 區段與相同 fee 下比較。  
- **Evaluation enhancement**：同時輸出 predictive metrics（Accuracy/F1）與 financial metrics（Return/Sharpe/MDD/Win Rate/Turnover）。

---

## 7) 常見錯誤與排查

### A) `scaler file not found`

原因：尚未先跑訓練，或 `--model` 路徑錯。  
解法：先確認 `model/<SYMBOL>/*.scaler.pkl` 是否存在，且與 `--model` 同 basename。

### B) `Insufficient rows for window`

原因：資料筆數不足或 `window` 太大。  
解法：降低 `--window`（如 30 -> 10）或下載更長期間資料（例如 5y）。

### C) `No CSV in ... for ticker ...`

原因：`historical_data` 中找不到對應檔名。  
解法：先跑 `get_stock_data.py`，並確認檔名形如 `AAPL_5y_1d.csv`。

### D) 指標結果波動很大

原因：樣本或 window 太小、epoch 太少。  
解法：固定 seed、增加資料長度、增加 epochs，並用 protocol 批次做平均比較。

---

## (1-minute) Quick Start 

在專案根目錄直接依序執行：

```powershell
uv sync
uv run python -m py_compile main.py get_stock_data.py train_stock.py predict_stock.py backtest_stock.py run_protocol.py
uv run get_stock_data.py --protocol experiment_protocol.json --window_idx 0
uv run train_stock.py --csv_dir historical_data --ticker AAPL --save_model dir_model.pt --window 30 --epochs 5 --protocol experiment_protocol.json
uv run predict_stock.py --csv historical_data/AAPL_5y_1d.csv --model model/AAPL/dir_model.pt
uv run backtest_stock.py --csv historical_data/AAPL_5y_1d.csv --model model/AAPL/dir_model.pt --protocol experiment_protocol.json --eval_split test
uv run run_protocol.py --protocol experiment_protocol.json --window_idxs 0 --epochs 1
```

Quick Start 產出重點：
- 模型與 artifacts：`model/AAPL/dir_model.pt`, `dir_model.scaler.pkl`, `dir_model.meta.json`
- 推論結果：`backtest_results/AAPL/AAPL_5y_1d_pred.csv`
- 回測結果：`backtest_results/AAPL/AAPL_5y_1d_bt_summary.json`
- 批次比較：`backtest_results/comparison_summary.csv`
