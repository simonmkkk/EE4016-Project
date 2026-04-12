## 簡報時間配置建議（約 10 分鐘）

| 報告區段 | 建議時間 | 對應課程重點 |
|------|----------|--------------|
| Part 1 問題定義與應用情境 | 約 1.5 分鐘 | 說明專案目標、目標使用者與實務價值 |
| Part 2 方法與工程設計 | 約 2 分鐘 | 交代系統流程、模組分工與可重現性設計 |
| Part 3 模型與理論基礎 | 約 2.5 分鐘 | 說明演算法核心（LSTM + 技術指標）與方法合理性 |
| Part 4 實驗與結果分析 | 約 2 分鐘 | 展示操作流程、回測指標與基準策略比較 |
| Part 5 局限與未來工作 | 約 1 分鐘 | 說明研究限制與後續延伸計畫 |



---

## 投影片逐頁細目（重寫版，對齊版本庫）

以下為「每個 Part 先總覽，再逐頁說明」；**全簡報共 29 頁（Page 1–29）**，頁碼連續，內容與 `README.md`、`train_stock.py`、`backtest_stock.py`、`run_protocol.py`、`experiment_protocol.json`、`app/paths.py` 一致。口頭講解可壓縮子 bullet，投影片只放關鍵字。

### Part 1 問題定義與應用情境（Page 1–7）

| 頁面 | 本頁主題 | 建議講解重點 |
|------|----------|--------------|
| Page 1 | 標題與成員 | 題目、課程、分工 |
| Page 2 | 市場難預測與古典指標 | 難點三項 + 指標能／不能做的事 |
| Page 3 | 三缺口與 Beat the Classics | 適應性／混合性／信任 + 公平對照 |
| Page 4 | 一句話專題與價值 | 輸入→模型→輸出→評估一氣呵成 |
| Page 5 | 端到端管線（Roadmap） | `experiment_protocol` + `run_protocol` 編排、目錄、防洩漏 |
| Page 6 | 系統工作流程（五步） | 下載→訓練→可選推論→回測對照→檢視產出；對齊插圖 |
| Page 7 | CLI 與可重現指令 | 僅命令列：腳本順序、關鍵參數、meta 指紋 |

#### Page 1 — 標題與成員
- **本頁重點**：第一句話說明「做什麼」，第二眼看到「誰負責什麼」。
- **講解細節**：
  - 標題建議：**Adaptive Multi-Granularity Neural Trading System**（與 `README.md` 一致），副標可寫「LSTM 方向預測 × 技術指標 × 回測對照」。
  - 交代：**課程名稱、學校、學期、報告日期**；若課程要求 Beat the Classics，可放一句「本報告對齊課程 beat 定義，結果以實際輸出檔為準」。
  - **成員區**：每人姓名、學號；**負責模組**建議對應真實分工（例如：資料與 protocol、模型與訓練、回測與圖表、簡報與文件）。
- **口頭範例（30 秒）**：「我們實作從 yfinance 到 LSTM 訓練與回測的完整管線，並在相同測試區段與手續費假設下，和 Buy-and-Hold 與 MACD／RSI／布林帶比較。」
- **建議視覺**：封面大字＋下方橫列頭像或姓名區；背景淺色、對比足。

#### Page 2 — 為何難預測？古典指標的角色？
- **本頁重點**：先承認**市場難預測**，再說明**為何仍要技術指標 + DL**。
- **講解細節**：
  - **難點**（可各一句）：**信噪比低**（價格近似隨機漫步成分）；**regime 切換**（趨勢／盤整／危機）；**摩擦與成本**（手續費、滑價，回測用 `--fee` 近似）。
  - **古典指標的價值**：RSI、MACD、布林帶等把價量壓成**可重複計算的規則**，便於當 **non-DL baseline**。
  - **限制**：**滯後**、**假訊號**、在極端行情可能集體失效——這正是「要 beat」的空間。
- **與程式對齊**：回測檔 `backtest_stock.py` 用 **ta** 計 MACD／RSI／布林，參數可經 `experiment_protocol.json` 的 `baseline_params` 與訓練時 `--protocol` 對齊。
- **建議視覺**：左欄三難點圖示；右欄「規則化 ✓／僵固 ✗」對照表。

#### Page 3 — 三大缺口與 Beat the Classics
- **本頁重點**：用**三缺口**收斂動機，用**Beat the Classics**收斂評估方式。
- **講解細節**：
  - **適應性**：希望模型從資料學到**非線性**，補足固定參數指標在 regime 變化時的弱點。
  - **混合性**：**OHLCV + 技術指標序列**一併進 LSTM（實作見 `train_stock.py` 的 `fe()`），不是純價格、也不是純規則。
  - **可信度**：以**時間切分、同一測試段、固定 fee** 做公平比較；口頭可提「我們不宣稱穩賺，只展示在約定協議下的相對表現」。
  - **Beat 框架**：**LSTM（model_lstm）** 對 **buy_and_hold、macd、rsi、bollinger**；協議檔定義 ticker 與資料視窗，`run_protocol.py` 批次產出 **`RESULTS_DIR/comparison_summary.csv`**（聚合多 ticker 後的摘要，與單次 `_bt_summary.json` 層級不同——口頭可一句帶過）。
- **建議視覺**：上排三缺口關鍵字；下排「我們 vs 四基準」圖示。

#### Page 4 — 專題一句話與價值主張
- **本頁重點**：聽眾離開這頁能複述：**吃什麼資料、用什麼模型、產出什麼結論**。
- **講解細節**：
  - **一句話版本**：「以 **yfinance** 下載之 **OHLCV** 建特徵（含 RSI／MACD／布林寬／ATR／量能與 **granularity**），**LSTM** 預測**下一棒漲跌方向**，再於 **test 區段**與基準策略做**相同 fee** 下的報酬與風險比較。」
  - **價值賣點（可選 3–5 個）**：可重現（`meta.json` 記錄指令與 git commit）、多視窗 protocol、可選 **Attention**、推論腳本含解釋欄位（`predict_stock.py`）。
- **建議視覺**：中央一條水平流程箭頭，上下點狀標籤列賣點。

#### Page 5 — 端到端管線（Roadmap）
- **本頁重點**：用**一張架構圖**說明「**設定檔 → 編排器 → 各腳本 → 產物目錄**」，並點出**時間序切分**與**與古典基準對照**；細節口頭呼應你投影片上的 **End-to-end pipeline (roadmap)**。
- **講解細節**（與版本庫一致，簡報若畫成 `record/`、`model/`、`result/` 可當圖示別名，**實際目錄名**以 `.env` 的 **`SAVE_DIR`／`MODEL_DIR`／`RESULTS_DIR`** 為準，預設為 `historical_data/`、`model/`、`backtest_results/`）：
  - **輸入設定**：**`experiment_protocol.json`**（ticker 宇宙、`data_windows`、切分比例、`baseline_params` 等）。
  - **編排器**：**`run_protocol.py`** 讀取 protocol，依序驅動下載 → 訓練 → 回測，並聚合出 **`RESULTS_DIR/comparison_summary.csv`**。
  - **資料下載**：**`get_stock_data.py`** → 原始 **OHLCV CSV** 寫入 **`SAVE_DIR`**。
  - **訓練**：**`train_stock.py`** → **`MODEL_DIR/<SYMBOL>/dir_model.pt`**，以及同 stem 的 **`.scaler.pkl`、`.meta.json`**（內含特徵欄位、window、`eval_threshold` 等）。
  - **回測與基準**：**`backtest_stock.py`** 載入模型與 scaler，在約定區段上比較 **LSTM（機率過 threshold 之訊號）** 與 **Buy-and-Hold、MACD、RSI、Bollinger**；產出明細 CSV、**`*_bt_summary.json`**。
  - **可選支線**：**`predict_stock.py`** 為**獨立推論**（含解釋欄位），**預設不在 `run_protocol.py` 的一鍵鏈內**，需手動執行。
  - **原則一句**：Train／Val／Test **依時間順序**，避免用未來資料；scaler **僅在 train 上 fit**（見 `train_stock.py`）；回測預設 **`--eval_split test`**。
- **建議視覺**：可直接使用你製作的 **Roadmap 投影片**（protocol → orchestrator → 三主流程 + predict 虛線可選）；或專案內 **`flowchart/flowchart.png`**。

#### Page 6 — 系統工作流程（How the System Works）
- **本頁重點**：用**五步故事線**帶聽眾走一遍系統——與 **Page 5** 分工如下：**Page 5** 偏「架構與編排」，**本頁**偏「從資料到結論的使用者視角」，邏輯與你附的 **五欄手繪／資訊圖**（*How the System Works (End-to-End Workflow)*）一致。
- **講解細節（五步，左→右）**：
  1. **DOWNLOAD DATA**  
     - **腳本**：`get_stock_data.py`。  
     - **產物**：依時間排序之 **OHLCV CSV**，落地 **`SAVE_DIR`**（圖中若標「雲端／API → 電腦」可保留）。  
     - **一句**：這裡只處理**原始價量**，尚未訓練。
  2. **TRAIN MODEL**  
     - **腳本**：`train_stock.py`（內含特徵工程 `fe()`、滑動視窗、LSTM 訓練）。  
     - **產物**：**`MODEL_DIR`** 下 **`dir_model.pt`** + **scaler** + **metadata（.meta.json）**——簡報圖中的 *artifacts* 盒裝即此三類。  
     - **一句**：模型與標準化參數綁定，供推論與回測共用。
  3. **（OPTIONAL）INFERENCE**  
     - **腳本**：`predict_stock.py`。  
     - **用途**：在**不做完整回測**時，仍可看**機率、解釋欄位**（*Standalone Predictions / Explanations*）。  
     - **一句**：與 **Page 5** 一致——**可選**；**不在** `run_protocol` 預設批次鏈必經步驟。
  4. **BACKTEST & COMPARE**  
     - **腳本**：`backtest_stock.py`。  
     - **內容**：同一測試區段上，**LSTM 策略** vs **Buy-and-Hold、MACD、RSI、Bollinger**；圖表可用多條曲線示意「我的模型 vs 基準」。  
     - **一句**：比較的是**策略報酬序列**（已納入 **`--fee`** 等假設），不是只報分類準確率。
  5. **REVIEW OUTPUTS**  
     - **目錄**：**`RESULTS_DIR`**（對照簡報上的「大資料夾」圖示）。  
     - **檔案**：各標的資料夾內 **回測明細 CSV**、**`*_bt_summary.json`**；若跑過 protocol 批次，根目錄另有 **`comparison_summary.csv`** 供**跨 ticker／視窗**對照。  
     - **一句**：這裡是報告「**有沒有 beat 基準**」的**證據落點**。
- **定位與免責（可選 10 秒）**：本系統適合**研究／課程展示**與**可重現實驗**，**非**投顧產品；不向聽眾提供買賣建議。
- **建議視覺**：直接使用你提供的 **五步插圖**（與本節標題逐欄對齊）；若投影片上資料夾名為 `record`／`result`，可加一行小字：**等同本專案 `SAVE_DIR`／`RESULTS_DIR`**。

#### Page 7 — CLI 指令與可重現性
- **本頁重點**：只透過**命令列**說明如何重跑流程；不介紹互動選單或其它入口。
- **講解細節**：
  - **建議執行方式**：在專案根目錄用 **`uv run python <腳本>.py ...`** 或 **`python <腳本>.py ...`**（與 `README` 一致）；不確定參數時先打 **`--help`**。
  - **典型管線（依序）**：
    1. **`get_stock_data.py`**：下載資料 → 寫入 `SAVE_DIR`。二擇一：**`--ticker` + `--interval` + `--years`**，或 **`--protocol <json>` + `--window_idx`**（對齊 `experiment_protocol.json` 的視窗）。
    2. **`train_stock.py`**：**`--csv_dir`**（或 `--csvs`）+ **`--ticker`** + **`--save_model`**；建議加 **`--protocol experiment_protocol.json`** 與切分一致；可選 **`--window`、`--epochs`、`--use_attn`**。
    3. **`backtest_stock.py`**：**`--csv`** + **`--model`**；可選 **`--protocol`**、**`--fee`**（預設 **0.001**）、**`--eval_split test|all`**、**`--threshold`**（省略則用模型 meta）。
    4. **可選**：**`predict_stock.py`**（推論與解釋欄位）；**`run_protocol.py --protocol ...`**（批次：內部連續呼叫下載→訓練→回測並寫 **`comparison_summary.csv`**）。
  - **可重現**：訓練產物旁的 **`.meta.json`** 含 **`reproducibility.command`**、**`git_commit`**、**`data_fingerprints`**；口頭一句「同一指令 + 同一資料指紋 → 可比較」。
- **建議視覺**：終端機截圖（可打馬賽克路徑）+ 一表三欄：**腳本｜關鍵 CLI 參數｜主要產出檔**。

---

### Part 2 方法與工程設計（Page 8–11）

| 頁面 | 本頁主題 | 建議講解重點 |
|------|----------|--------------|
| Page 8 | 資料來源與格式 | yfinance、interval 上限、CSV 欄位 |
| Page 9 | 特徵工程 | 動能／波動／量能／K 線形態 + 標準化 |
| Page 10 | Granularity 與 protocol 視窗 | 旗標定義 + `experiment_protocol.json` |
| Page 11 | 標籤定義 | `log_ret`、`direction`、與視窗對齊 |

#### Page 8 — 資料來源與格式
- **本頁重點**：資料**從哪裡來**、**長什麼樣子**，避免聽眾質疑可重現性。
- **講解細節**：
  - **來源**：**yfinance**（`get_stock_data.py`），下載後寫入 **`SAVE_DIR`**。
  - **結構**：時間索引 **`date`**（parse 為 datetime）、欄位 **OHLCV**；**依時間排序**。
  - **interval**：支援多種（腳本內有 `INTERVAL_LIMITS`）；**不同 interval 有不同最大回溯**，口頭一句即可。
  - **Protocol 模式**：讀取 `experiment_protocol.json` 的 **`ticker_universe`** 與 **`data_windows`**（預設含 **`1d` 約 5 年**與 **`1h` 約 2 年** 兩個視窗），用 **`--window_idx`** 選第幾個視窗下載。
- **建議視覺**：CSV 前五列示意（遮 ticker）；旁註「date 升冪」。

#### Page 9 — 特徵工程（對齊 `train_stock.fe()`）
- **本頁重點**：**模型實際吃進去的欄位**與**為何要標準化**。
- **講解細節**：
  - **動能**：**RSI**、`macd`（MACD **diff**）、**布林帶寬 `bbw`**。
  - **波動**：**ATR**、布林寬。
  - **量能**：**vma20**（量之 20 期 SMA）、**v_ratio**（量／vma20）。
  - **價格形態（勿漏）**：**body**（開收價差絕對值）、**range**（高低價差）——簡報若只列傳統指標會與程式不符。
  - **Granularity**：由時間戳是否「非全日 00:00:00」判斷 **0=日類、1=含時分秒之 bar**（見 `read_and_fe()`），並納入特徵。
  - **標準化**：**`StandardScaler` 只在 train split 的合併資料上 `fit`**，再 transform train／val／test，避免洩漏。
  - **缺值**：指標暖機後 **`dropna()`**。
- **建議視覺**：四象限或心智圖「動能／波動／量能／形態+粒度」。

#### Page 10 — Granularity 與多時間尺度（含 protocol）
- **本頁重點**：**為何同一模型能混多頻率**，以及**實驗預設做了什麼**。
- **講解細節**：
  - **概念**：不同 **bar** 代表不同資訊密度；**granularity 特徵**讓模型知道「這列來自日線還是較細的 bar」。
  - **實作**：`granularity` 為 **0/1**；若同一模型餵多個 CSV，`train_stock.py` 會 **concat** 多序列建樣本（`build_seq_multi`）。
  - **與 `experiment_protocol.json` 對齊**：預設 **`data_windows`** 同時包含 **日線與小時線** 視窗（非「紙上多頻率」）；報告可說明實際展示的是**其中一個視窗**還是**兩者都跑**。
  - **延伸口頭**：分鐘線可下載但受 yfinance 回溯長度限制——與 `INTERVAL_LIMITS` 一致。
- **建議視覺**：兩條時間軸（1d vs 1h）+ 同一模型圖示；旁列 protocol 兩個 window。

#### Page 11 — 標籤定義（與程式逐字一致）
- **本頁重點**：**標籤定義錯，後面全部不成立**。
- **講解細節**：
  - 在 `fe()` 中：`log_ret = log(close).diff().shift(-1)`，即**當前棒對應的「下一期」對數報酬**。
  - **`direction = (log_ret > 0)`**，**1** 表下一期上漲、**0** 表下跌／持平（嚴格說 `log_ret > 0` 才為 1）。
  - **滑動視窗對齊**：`build_seq` 在索引 `i` 取 `[i-window, i)` 的特徵，標籤為 **`direction[i]`**——即**以第 i 根棒為「當前」時，預測下一棒方向**。
  - **為何用對數報酬**：尺度較穩、便於與序列模型搭配（口頭一句）。
- **建議視覺**：小表格三列：t、t+1、log_ret 正負 → direction。

---

### Part 3 模型與理論基礎（Page 12–18）

| 頁面 | 本頁主題 | 建議講解重點 |
|------|----------|--------------|
| Page 12 | 任務形式 | 序列二元分類、維度、與回測訊號差異 |
| Page 13 | LSTM 架構 | 單層 LSTM、128、Attention、logit |
| Page 14 | 損失與類別權重 | BCEWithLogitsLoss、自動 pos_weight |
| Page 15 | 時間切分與防洩漏 | 70/15/15、protocol 覆寫、早停 |
| Page 16 | 推論、meta、訊號 | scaler、threshold；LSTM 僅 ±1 |
| Page 17 | 基準策略 | 四基準規則要點、參數來自 protocol |
| Page 18 | 回測與損益 | 報酬公式、fee、Sharpe、MDD |

#### Page 12 — 任務形式
- **本頁重點**：**訓練任務**與**回測訊號**不要混成一種「動作空間」。
- **講解細節**：
  - **輸入**：張量形狀 **`(batch, window, n_features)`**，`n_features` 含 **granularity** 等欄位。
  - **輸出**：單一 **logit**（再接 sigmoid 得機率）。
  - **訓練標籤**：二元 **0/1**。
  - **回測時 LSTM 訊號**（見 `backtest_stock.py`）：`prob > threshold` → **+1（多）**，否則 **-1（空）**——**沒有 0（空手）**；這點與部分古典策略不同，**Page 17** 會對照。
- **建議視覺**：滑動視窗圖 + 小註「最後一步對齊 direction[i]」。

#### Page 13 — LSTM 架構（`LSTMDir`）
- **本頁重點**：**夠具體**讓老師知道你不是套黑箱。
- **講解細節**：
  - **單層 LSTM**，**batch_first=True**，**hidden 預設 128**（與 `train_stock.py` 一致）。
  - **Attention（可選 `--use_attn`）**：對每步 hidden 算權重 softmax，加權和後送 **FC**；未開則取**最後一步** `o[:, -1]`。
  - **輸出層**：`nn.Linear(hid, 1)` → **logit**。
- **建議視覺**：方塊圖「Features → LSTM → [Attn] → FC → logit」。

#### Page 14 — 損失函數與類別不平衡
- **本頁重點**：與實作一致——**pos_weight 由訓練集自動算**，不是「有時才開」。
- **講解細節**：
  - **損失**：`BCEWithLogitsLoss`。
  - **pos_weight**：依訓練集 **正樣本比例** 設為 **`neg_ratio / pos_ratio`**（程式用 `safe_pos` 避免除零），**每個訓練 run 必算**。
  - **驗證**：epoch 迴圈中印 **train/val loss**；**early stopping** 依 **val loss** + **`patience`**（預設 6）。
  - **指標**：`evaluate_split` 輸出 **Accuracy、Precision、Recall、F1**；使用 **`eval_threshold`**（預設 0.5）將機率二值化。
- **建議視覺**：小公式 `pos_weight = N_neg / N_pos`（概念）+ loss 曲線示意。

#### Page 15 — 時間切分、protocol 與防洩漏
- **本頁重點**：**這組專題可信度的核心**。
- **講解細節**：
  - **預設比例**：**train 70%／val 15%／test 15%**（`train_ratio`、`val_ratio`，餘為 test）。
  - **Protocol**：若 `--protocol` 且 `split.method == "ratio"`，則 **train_ratio／val_ratio 由 JSON 覆寫**（與 `experiment_protocol.json` 一致）。
  - **順序切分**：**時間由早到晚**，不可打亂；**超參**（如是否 early stop）看 **val**，**最終敘事**以 **test** 為主（與 `backtest_stock.py --eval_split test` 一致）。
  - **可選**：`--walk_forward_folds` 在 **test 段**做滾動折數評估（**有參數、非預設 protocol 必跑**）；口頭可說「屬延伸實驗」。
- **建議視覺**：時間軸三色 Train／Val／Test；箭頭「資訊只往未來流」。

#### Page 16 — 推論、meta 與交易訊號
- **本頁重點**：**機率如何變部位**，以及 **fee 從哪裡來**。
- **講解細節**：
  - **載入**：模型權重 + **`.scaler.pkl`** + **`.meta.json`**（內含 `features`、`window`、`eval_threshold`、`use_attn`）。
  - **Threshold**：`backtest_stock.py` 若未傳 `--threshold`，則用 **meta 的 `eval_threshold`**；與訓練時評估閾值一致。
  - **LSTM 訊號**：**僅 +1 / -1**（見 Page 12）；**沒有 flat**。
  - **手續費**：`--fee` **每次部位變化**計價，預設 **0.001（0.1%）**；與 `strategy_metrics` 中 `turnover` 一致。
- **建議視覺**：機率條 + threshold 豎線；下方註「LSTM：多或空」。

#### Page 17 — 基準策略（對齊 `backtest_stock.py` + protocol）
- **本頁重點**：**每個 baseline 的直覺規則**與**參數可從 protocol 讀**。
- **講解細節**：
  - **Buy-and-Hold**：訊號全 **+1**。
  - **MACD**：**macd_line > macd_signal** → +1，否則 -1（**二態**）。
  - **RSI**：低於 **oversold** → +1；高於 **overbought** → **-1**；否則 **0**（**可空手**）。
  - **布林帶**：收盤 **< 下軌** → +1；**> 上軌** → -1；否則 **0**。
  - **參數**：預設與 `experiment_protocol.json` 的 **`baseline_params`** 一致（MACD 快慢、RSI 窗與閾值、布林窗與倍數）；`--protocol` 可載入自訂。
  - **勿宣稱未實作基準**：版本庫**無 OBV baseline**；口頭與投影片請勿列入「已比較」清單。
- **建議視覺**：四列小圖示 +「二態 vs 三態」對照一行字。

#### Page 18 — 回測邏輯與財務指標（`backtest_stock.py`）
- **本頁重點**：**報酬怎麼疊加**、**為何能輸出 Sharpe／MDD**。
- **講解細節**：
  - **報酬對齊**：特徵窗對齊後，用 **`fwd_ret`（下一期百分比報酬）** 與**訊號**相乘得策略單期報酬。
  - **成本**：`strat_ret = signal * fwd_ret - fee * turnover`，**turnover** 為部位變化絕對值（含 0→±1、±1→±1）。
  - **累積**：`equity = cumprod(1 + strat_ret)`。
  - **指標**：**total_return**、**Sharpe**（依檔名推 **年化期數** `periods_per_year`）、**max_drawdown**；另印 **win_rate、換手**等。
  - **輸出**：明細 CSV + **`_bt_summary.json`**；含 **predictive_metrics_model**（將 LSTM 多／空與**真實漲跌**比較的 accuracy／F1）。
- **建議視覺**：簡式現金流時間軸 + 公式一行。

---

### Part 4 實驗與結果分析（Page 19–24）

| 頁面 | 本頁主題 | 建議講解重點 |
|------|----------|--------------|
| Page 19 | 技術棧與環境 | PyTorch、ta、uv、.env |
| Page 20 | Demo 與結果從哪來 | 選單 vs CLI vs protocol |
| Page 21 | 分類指標 | 四率 + threshold |
| Page 22 | 財務指標 | 報酬、Sharpe、MDD、解讀 |
| Page 23 | 是否 beat 基準 | 對照 protocol 輸出、誠實結論 |
| Page 24 | 視覺化 | 權益曲線、長條圖 |

#### Page 19 — 技術棧與可重現環境
- **本頁重點**：**跟得上課程的「實作」要求**。
- **講解細節**：
  - **核心**：**PyTorch**（LSTM）、**yfinance**、**pandas/numpy**、**ta**（指標）、**scikit-learn**（Scaler、指標）。
  - **專案管理**：**`pyproject.toml` + uv**（`uv sync`／`uv run`），較易固定版本。
  - **路徑**：**`.env`** 設定 **`SAVE_DIR`／`MODEL_DIR`／`RESULTS_DIR`**（見 `app/paths.py`）。
- **建議視覺**：logo 拼貼或清單；角落小字「uv + .env」。

#### Page 20 — Demo 流程與產出物
- **本頁重點**：老師聽得懂「**你現場要點哪裡**」。
- **講解細節**：
  - **路線 A（互動）**：`uv run main.py` → **5 Pipeline** 跑 **`run_protocol.py`**（或逐步 1→2→3）。
  - **路線 B（課堂最短路）**：已備份 CSV／模型時，直接 **Backtest** 或 CLI **`backtest_stock.py`**。
  - **路線 C（完整故事）**：強調 **`comparison_summary.csv`** 是 **protocol 批次**後、**跨 ticker 聚合**的表——適合當「**是否 beat**」的**摘要證據**；單次回測看 **`_bt_summary.json`**。
  - **展示技巧**：螢幕放大終端字體；預先確認 **`RESULTS_DIR`** 有最新檔。
- **建議視覺**：三分支流程圖標「時間夠用哪條」。

#### Page 21 — 分類結果指標
- **本頁重點**：**先談預測**，再談賺錢——邏輯與口試一致。
- **講解細節**：
  - 來源：**訓練結束印出 [VAL]／[TEST]**；**回測 summary** 另有 **predictive_metrics_model**（訊號方向 vs 真實漲跌）。
  - **四項**：Accuracy、Precision、Recall、F1；**說明 imbalance 時 F1 比 accuracy 有參考價值**。
  - **Threshold**：提高 threshold 可能 **precision ↑、recall ↓**——一句帶過即可。
- **建議視覺**：單表或雷達圖擇一；註明「TEST」字樣。

#### Page 22 — 財務結果指標
- **本頁重點**：把 **Sharpe、MDD** 講清楚，避免只唸數字。
- **講解細節**：
  - **total_return**：策略累積報酬（已扣 **fee** 影響之單期報酬序列）。
  - **Sharpe**：假設報酬近似平穩，**年化**（程式用推估之 **periods_per_year**）；說明「僅歷史樣本估計」。
  - **max_drawdown**：最大回撤，**風險**指標；可與報酬並列討論。
  - **引用**：單次結果用 **`_bt_summary.json` 的 strategies**；多標的摘要用 **`comparison_summary.csv`** 的 **avg_*** 欄位——**勿混成同一層級**。
- **建議視覺**：表頭含 **model_lstm vs baselines** 五列。

#### Page 23 — 我們有沒有 beat 基準？
- **本頁重點**：**結論明確**，並符合課程對 **beat** 的期待。
- **講解細節**：
  - **定義**：在**同一測試段、同一 fee、同一組 baseline 參數**下，比較 **total_return／Sharpe／MDD**（簡報選 1–2 個主指標即可，**勿一口氣念全部**）。
  - **證據**：指到 **`comparison_summary.csv`** 或投影片上的表（若課堂不能開 repo，事先匯出圖表）。
  - **若未 beat**：**誠實**——可能原因：**市場效率、樣本外 regime、成本、指標／架構仍簡化**；強調**公平對照已完成**仍具專題價值。
  - **若部分勝出**：說明**哪個視窗／哪個標的**，避免過度泛化。
- **建議視覺**：紅綠表或「主指標冠軍次數」小柱狀圖。

#### Page 24 — 視覺化（建議保留）
- **本頁重點**：一張圖一個訊息。
- **講解細節**：
  - **權益曲線**：來自回測明細 CSV 的 **`equity`** 欄位；可與 **Buy-and-Hold** 的累積路徑對照（需自行從 `bh_ret` 累積或簡化圖）。
  - **長條圖**：五策略 **total_return** 並列，最直觀。
  - **避免**：同頁塞滿子圖；字體要大於 18pt 以利投影。
- **建議視覺**：全寬單圖 + 來源註腳（ticker、interval、test）。

---

### Part 5 局限與未來工作（Page 25–27）

| 頁面 | 本頁主題 | 建議講解重點 |
|------|----------|--------------|
| Page 25 | 研究局限 | 資料、泛化、因果、成本 |
| Page 26 | 未來工作 | 與現況銜接（含已部分實作者） |
| Page 27 | Thank You | 致謝 |


#### Page 25 — 研究局限性
- **本頁重點**：**主動揭露**，加分項目。
- **講解細節**：
  - **資料**：yfinance **非即時撮合資料**；**倖存者偏差**未處理；**除權息**簡化。
  - **泛化**：單一或少量 ticker、**未涵蓋所有市場狀態**。
  - **因果**：統計相關 **≠** 可交易 alpha；**過擬合**風險永遠存在。
  - **摩擦**：**fee** 為固定比例；**滑價、衝擊、流動性**未細建模。
- **建議視覺**：四點清單，每點不超過一行字。

#### Page 26 — 未來工作（與版本庫現況銜接）
- **本頁重點**：**可執行**、**分短中長期**，並避免「其實已經有一點」卻寫成從零開始。
- **講解細節**：
  - **短期**：**系統化 walk-forward**（目前已有 **`--walk_forward_folds`**，可擴成完整報告與圖表）；**多標的**已在 protocol 架構下，可擴**產業／市場**。
  - **中期**：**成本模型**（滑價、衝擊）、**threshold 與 fee 敏感度**表格。
  - **長期**：**SHAP／解釋**與 `predict_stock.py` 的解釋欄位深化；**指標參數搜尋／演化式**優化（提案曾寫，可標「尚未納入主幹」若屬實）。
- **建議視覺**：時間軸三格；每格 2 個 bullet。

#### Page 27 — Thank You

---
