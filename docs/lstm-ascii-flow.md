# LSTM 完整流程：輸入 -> 過程 -> 輸出（ASCII Art）

---

## 一、高層總覽

```
+------------------+     +--------------------------------------------+     +------------------+
|      Input       |     |              LSTM  Process                 |     |      Output      |
|                  |     |                                            |     |                  |
|  price + techs   +---->+  cell_1 --> cell_2 --> ... --> cell_T      +---->+  direction logit |
|  (normalized)    |     |                                            |     |  0=Down  1=Up    |
+------------------+     +--------------------------------------------+     +------------------+
  (batch, T, 14)              h, c  state passes between time steps            logit -> sigmoid
```

- 輸入層：原始行情 + 技術指標，已 StandardScaler 標準化，形狀 `(batch, T, d_in)`
- LSTM 層：沿 T 個時間步展開，h/c 狀態在步間傳遞
- 輸出層：取最後步 H(T)，接全連接層得到方向 logit

---

## 二、輸入層細節

```
  sliding window = T K-bars  (StandardScaler normalized)

  time axis ----------------------------------------------------------------->
              t=1          t=2          t=3      ...       t=T
               |            |            |                  |
               v            v            v                  v
           +--------+   +--------+   +--------+         +--------+
           |   X1   |   |   X2   |   |   X3   |   ...   |   XT   |
           +--------+   +--------+   +--------+         +--------+
            14 feats      14 feats     14 feats            14 feats
               |            |            |                  |
               +------------+------------+------------------+
                                     |
                             shape: (batch, T, 14)

  14 input features  (d_in = 14):
  +------------------------------------------------------------------+
  |  open    high    low    close    volume    granularity           |
  |  rsi     macd    bbw    atr      vma20     v_ratio   body  range |
  +------------------------------------------------------------------+
  * log_ret and direction are EXCLUDED (they are labels, not inputs)
```

---

## 三、LSTM 細胞：單一時間步（t 步）

### 3a. 輸入串接與門控計算

```
  X(t) -----+
  H(t-1) ---+-----> [X(t) ; H(t-1)]  (concat)
                            |
             +--------------+--------------+--------------+
             |              |              |              |
             v              v              v              v
          +------+       +------+       +------+       +------+
          |  F   |       |  I   |       |  C~  |       |  O   |
          | sig  |       | sig  |       | tanh |       | sig  |
          +--+---+       +--+---+       +--+---+       +--+---+
             |              |              |              |
            f_t            i_t           C~_t            o_t
```

```
  F  Forget Gate  遺忘門   f_t  = sigma( Wf * [H(t-1); X(t)] + bf )
  I  Input  Gate  輸入門   i_t  = sigma( Wi * [H(t-1); X(t)] + bi )
  C~ Candidate    候選值   C~_t = tanh(  Wc * [H(t-1); X(t)] + bc )
  O  Output Gate  輸出門   o_t  = sigma( Wo * [H(t-1); X(t)] + bo )
```

### 3b. 細胞狀態更新

```
             f_t                   i_t   C~_t
              |                     |     |
  C(t-1) --> [*] --> f_t*C(t-1) --> [+] <-+-- i_t*C~_t
                                     |
                                   C(t)  <-- new cell state (long-term memory)
                                     |
                                   tanh
                                     |
                      o_t -------> [*]
                                     |
                                   H(t)  <-- new hidden state (output)
```

```
  Update formulas:
    C(t) = f_t * C(t-1)  +  i_t * C~_t
    H(t) = o_t * tanh( C(t) )
```

---

## 四、序列展開：T 步串接

```
  h0=0                  h1                   h2                      hT
  c0=0                  c1                   c2                      cT
    |                    |                    |                        |
    v                    v                    v                        v
  +--------+  h1,c1  +--------+  h2,c2  +--------+         +--------+
  | cell_1 | ------> | cell_2 | ------> | cell_3 |  . . .  | cell_T |
  +--------+         +--------+         +--------+         +----+---+
      ^                   ^                  ^                   |
      |                   |                  |                  H(T)  <- take last step
     X1                  X2                 X3                   |
                                                                 v
                                                          +------------+
                                                          |   Linear   |
                                                          |  hid --> 1 |
                                                          +-----+------+
                                                                |
                                                             logit
```

---

## 五、輸出層細節

```
                                      +----------------------------------+
  H(T)      +------------------+      | logit (scalar)                   |
  +-----+   |                  |      |                                  |
  | hid +-->|  fc: hid -> 1    +----->| [Train]                          |
  +-----+   |                  |      |   BCEWithLogitsLoss              |
            +------------------+      |   vs. direction label (0 or 1)   |
                                      |                                  |
                                      | [Infer]                          |
                                      |   sigmoid(logit) = P(Up)         |
                                      |   >= 0.5  =>  predict Up   (1)   |
                                      |   <  0.5  =>  predict Down (0)   |
                                      +----------------------------------+
```

---

## 六、端到端完整流程（一條樣本）

```
  raw CSV
      |
      v
  fe() ---> add: rsi, macd, bbw, atr, vma20, v_ratio, body, range
      |
      v
  StandardScaler  (fit on train split only, then transform all splits)
      |
      v
  build_seq (sliding window)
      |
      +----> X : (batch, T, 14)
      +----> y : (batch,)          direction of next bar
      |
      v
  t=1    X1 ---> [cell] --+
  t=2    X2 ---> [cell] --+
   ...                    +----> H(T) ---> Linear(hid, 1) ---> logit
  t=T    XT ---> [cell] --+                                        |
                                                                   v
                                              BCEWithLogitsLoss( logit, y )
                                                                   |
                                                             .backward()
                                                             optimizer.step()
```

```
  Dimension summary:
  +--------------------+----------+-----------------------------+
  |  Symbol            |  Value   |  Meaning                    |
  +--------------------+----------+-----------------------------+
  |  d_in              |  14      |  input feature count        |
  |  T  (window)       |  30      |  time steps per sample      |
  |  hid               |  128     |  LSTM hidden size           |
  |  batch             |  64      |  samples per batch          |
  |  output            |  1       |  direction logit            |
  +--------------------+----------+-----------------------------+
```

---

## 相關文件

- [LSTM 細胞內部機制](./lstm-cell-internals.md)
- [訓練資料與狀態示例](./train-sequence-state-example.md)
- [MLP 與 LSTM 概念比較](./mlp-lstm-concept-examples.md)
