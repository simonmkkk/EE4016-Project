# 訓練資料與 LSTM 狀態：玩具例子

本文用**極小假資料**對照 `train_stock.py` 的流程，說明以下四件事：

1. `FEATS` 的來源與欄位組成
2. 滑動視窗如何生成 `(batch, T, d_in)` 張量
3. 單條序列在 `nn.LSTM` 內部的 hidden / cell 狀態傳遞
4. Batch 內多條序列的狀態隔離

欄位名稱與程式一致；**數值皆為示意**（僅說明形狀與流程，不代表真實行情）。

---

## 1. 輸入欄位：`FEATS` 的定義

### 1.1 程式定義

`read_and_fe()` 讀 CSV 後經 `fe()` 產生技術指標，`FEATS` 取**除了** `date`、`log_ret`、`direction`、`series_id` 以外的所有欄位（`granularity` 確保在內）：

```python
FEATS = [c for c in frames[0].columns if c not in ["date", "log_ret", "direction", "series_id"]]
if "granularity" not in FEATS:
    FEATS.append("granularity")
```

### 1.2 `fe()` 產生的衍生欄位

```python
def fe(df: pd.DataFrame):
    df = df.copy()
    df["rsi"]     = RSIIndicator(df["close"]).rsi()
    df["macd"]    = MACD(df["close"]).macd_diff()
    df["bbw"]     = BollingerBands(df["close"]).bollinger_wband()
    df["atr"]     = AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
    df["vma20"]   = SMAIndicator(df["volume"], 20).sma_indicator()
    df["v_ratio"] = df["volume"] / df["vma20"]
    df["body"]    = (df["open"] - df["close"]).abs()
    df["range"]   = df["high"] - df["low"]
    df["log_ret"]   = np.log(df["close"]).diff().shift(-1)
    df["direction"] = (df["log_ret"] > 0).astype(np.float32)
    return df.dropna().reset_index(drop=True)
```

> `log_ret` 和 `direction` 由 `fe()` 寫入，但**不**進 `FEATS`（前者是目標的基礎，後者是標籤）。

### 1.3 欄位一覽（典型 `d_in = 14`）

| 欄位 | 意義 |
|------|------|
| `open` / `high` / `low` / `close` / `volume` | 來自行情 CSV |
| `granularity` | 日線為 `0`，含非零時分秒則為 `1`（分鐘級） |
| `rsi` | 相對強弱指標 |
| `macd` | MACD 差值（`macd_diff`） |
| `bbw` | 布林帶寬 |
| `atr` | 平均真實區間 |
| `vma20` | 成交量 20 期均線 |
| `v_ratio` | 成交量 / `vma20` |
| `body` | \|開盤 − 收盤\| |
| `range` | 高 − 低 |

`d_in = len(FEATS) = 14`，對應 `LSTMDir(len(FEATS), …)` 的第一個參數。

---

## 2. 示意資料表

訓練前會用 **`StandardScaler`（只在 train 上 `fit`）** 再對各 split `transform`。下表為 **transform 後**的示意數值，並附標籤 `direction`（**不**進模型輸入），設 5 個時間列：

| `t` | `open` | `high` | `low` | `close` | `volume` | `gran.` | `rsi` | `macd` | `bbw` | `atr` | `vma20` | `v_ratio` | `body` | `range` | `direction` |
|----:|-------:|-------:|------:|--------:|---------:|--------:|------:|-------:|------:|------:|--------:|----------:|-------:|--------:|:-----------:|
| 0 | 0.02 | 0.05 | −0.01 | 0.03 | 0.10 | 0 | −0.20 | 0.04 | 0.00 | 0.06 | 0.08 | 0.12 | 0.01 | 0.02 | 1 |
| 1 | 0.01 | 0.02 | −0.02 | 0.00 | −0.05 | 0 | 0.10 | −0.03 | 0.01 | 0.02 | 0.07 | −0.08 | 0.00 | 0.01 | 0 |
| 2 | 0.03 | 0.04 | 0.01 | 0.04 | 0.20 | 0 | 0.25 | 0.10 | 0.02 | 0.05 | 0.09 | 0.15 | 0.02 | 0.01 | 1 |
| 3 | −0.10 | 0.00 | −0.08 | −0.06 | 0.05 | 0 | 0.05 | −0.12 | 0.03 | 0.04 | 0.10 | 0.02 | 0.04 | 0.06 | 0 |
| 4 | 0.06 | 0.08 | 0.02 | 0.07 | 0.11 | 0 | −0.05 | 0.08 | 0.01 | 0.03 | 0.11 | 0.06 | 0.01 | 0.02 | 1 |

以下例子設 **`window = 3`**（`T = 3`）以便圖解；實際訓練常用 `--window 30`。

---

## 3. 滑動視窗 → 樣本 `X` 與標籤 `y`

邏輯與 `build_seq` 相同：對每個 `i = window, …, len-1`，取 `v[i-window : i]` 為輸入，`d[i]` 為標籤。

```text
列索引:   t=0   t=1   t=2   t=3   t=4
           ┌─────┬─────┬─────┬─────┬─────┐
           │     │     │     │     │     │
           └─────┴─────┴─────┴─────┴─────┘
                 ╰──── 樣本 A ────╯  ↑ 預測 t=3 的 direction（= 0）
                       ╰──── 樣本 B ────╯  ↑ 預測 t=4 的 direction（= 1）
```

| 樣本 | 輸入列 | 輸入形狀 | 標籤 |
|:----:|:------:|:--------:|:----:|
| A | t = 0, 1, 2 | `(3, 14)` | `direction[3]` = **0** |
| B | t = 1, 2, 3 | `(3, 14)` | `direction[4]` = **1** |

將 A、B 疊成一個 batch（`batch = 2`），張量 `xb` 的形狀為 **`(2, 3, 14)`** = `(batch, T, d_in)`。

展開樣本 A 的輸入序列：

```text
  t=0  x[0] = [ open, high, low, close, volume, gran., rsi, macd, bbw, atr, vma20, v_ratio, body, range ]
  t=1  x[1] = [ ...14 維... ]
  t=2  x[2] = [ ...14 維... ]
              ╰──────────────────────── (3, 14) = (T, d_in) ──────────────────────╯
  標籤 y_A = direction[3] = 0
```

---

## 4. 單條序列的 `forward`：LSTM 內部狀態流

`nn.LSTM` 在**同一條樣本內**，依序處理 `t = 0 → 1 → 2`。未手動傳入初始狀態時，`h_0 = c_0 = 0`。

```text
  ┌───────────────────────────────────────────────┐
  │  樣本 A，序列長度 T = 3（batch 維先忽略）       │
  └───────────────────────────────────────────────┘

   x[0]           x[1]           x[2]
    │               │               │
    ▼               ▼               ▼
  ┌──────┐        ┌──────┐        ┌──────┐
  │ LSTM │──────► │ LSTM │──────► │ LSTM │
  │ cell │        │ cell │        │ cell │
  └──┬───┘        └──┬───┘        └──┬───┘
     │               │               │
  h_0, c_0        h_1, c_1        h_2, c_2
  （初始為 0）   （來自前一步）   （來自前一步）
     │               │               │
    o_0             o_1             o_2   ← 輸出張量 (1, 3, hid)
```

**未開 attention** 時，`LSTMDir` 只取**最後一步** `o[:, -1]`，再經 `fc` 得到一個 logit：

```text
  o_0 ─┐
  o_1 ─┤  捨棄
  o_2 ─┘──► Linear(hid → 1) ──► logit ──► BCEWithLogitsLoss vs y_A
```

---

## 5. Batch 並行：狀態不跨樣本混用

同一個 `forward(xb)` 中，`xb` 形狀 **`(2, 3, 14)`**，A 和 B 各自維護獨立的 `h, c` 鏈：

```text
  ┌──────────────────────┐    ┌──────────────────────┐
  │  序列 A              │    │  序列 B              │
  │  t=0 ──► t=1 ──► t=2 │    │  t=0 ──► t=1 ──► t=2 │
  │  h,c 僅在 A 內傳遞   │    │  h,c 僅在 B 內傳遞   │
  └──────────┬───────────┘    └──────────┬───────────┘
             │                           │
             ▼                           ▼
          logit_A                     logit_B
             │                           │
             └─────────────┬─────────────┘
                           ▼
                BCEWithLogitsLoss（對 batch 內所有樣本一起算）
```

A 的最後狀態**不會**成為 B 的初始狀態。這是 `nn.LSTM` 的預設行為，也符合「每個滑動視窗是獨立訓練樣本」的設計。

---

## 6. Backprop（直覺圖）

```text
  loss
   │
   ▼
  fc  （梯度回到 o_2，或 attention 加權後的向量）
   │
   ▼
  LSTM 時間反向（BPTT）：  t=2 ◄── t=1 ◄── t=0
   │
   ▼
  更新 LSTM 與 fc 的權重
  （FEATS 本身的計算公式固定在 fe() 裡，不更新）
```

---

## 7. 與 `train_stock.py` 的對應速查

| 概念 | 玩具例子 | 程式對應 |
|------|----------|----------|
| 每步輸入向量維度 | `d_in = 14` | `d_in = len(FEATS)` |
| 序列長度 | `T = 3` | `T = args.window` |
| 一批樣本數 | `batch = 2` | `args.batch`（DataLoader） |
| 標籤來源 | 視窗後一天的 `direction` | `build_seq` 的 `y` |
| 損失函數 | — | `BCEWithLogitsLoss` |
| 跨樣本狀態 | 不傳遞 | `nn.LSTM` 預設，不跨樣本保留 |

---

> 本文件僅輔助理解形狀與狀態流。實際訓練以程式為準，模型使用的欄位以 `model.meta.json` 中的 `features` 為最終依據。
