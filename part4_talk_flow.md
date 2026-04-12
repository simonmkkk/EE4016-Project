# Part 4: Experiment and Results

```
+---------------------------------------------------------------+
|  SLIDE 1 -- Setup                                             |
|                                                               |
|  Message: "Here is what makes our comparison fair."          |
|                                                               |
|  +---------------+  +---------------+  +-------------------+ |
|  | Test Segment  |  | Cost Setting  |  | 5 Strategies      | |
|  |               |  |               |  |                   | |
|  | last 15%      |  | fee = 0.1%    |  | LSTM              | |
|  | of timeline   |  | per position  |  | Buy-and-Hold      | |
|  | (no leakage)  |  | change        |  | MACD              | |
|  +---------------+  +---------------+  | RSI               | |
|                                        | Bollinger         | |
|                                        +-------------------+ |
|                                                               |
|  Key line: "Same window. Same fee. Five strategies. Fair."   |
+---------------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------------+
|  SLIDE 2 -- Numbers                                           |
|                                                               |
|  Message: "Here are the actual numbers we got."              |
|                                                               |
|  +---------------------------+  +---------------------------+ |
|  | Classification Metrics    |  | Financial Metrics         | |
|  | (how well it predicts)    |  | (how well it trades)      | |
|  |                           |  |                           | |
|  | Accuracy                  |  | Total Return              | |
|  | F1 Score  <-- focus here  |  | Sharpe Ratio              | |
|  | Precision                 |  | Max Drawdown              | |
|  | Recall                    |  |                           | |
|  |                           |  | --> table: 5 strategies   | |
|  | source: TEST split        |  | source: bt_summary.json   | |
|  +---------------------------+  +---------------------------+ |
|                                                               |
|  Key line: "Predict first, then check if it makes money."    |
+---------------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------------+
|  SLIDE 3 -- Verdict                                           |
|                                                               |
|  Message: "One chart. One conclusion. Did we beat them?"     |
|                                                               |
|  +---------------------------------------------+             |
|  |  Equity Curve  OR  Bar Chart (pick one)     |             |
|  |                                             |             |
|  |  LSTM        ----------                     |             |
|  |  BnH         . . . . .                      |             |
|  |  MACD        - - - - -                      |             |
|  |                                             |             |
|  |  label: ticker / interval / test segment   |             |
|  +---------------------------------------------+             |
|                                                               |
|  +---------------------------------------------+             |
|  | Conclusion (pick one):                      |             |
|  |                                             |             |
|  |  [WIN]     LSTM Sharpe > all 4 baselines    |             |
|  |            on ticker X, test segment        |             |
|  |                                             |             |
|  |  [PARTIAL] Daily: win / Hourly: loss        |             |
|  |            reason: regime shift             |             |
|  |                                             |             |
|  |  [LOSS]    Fair setup complete.             |             |
|  |            Limitations covered in Part 5.  |             |
|  +---------------------------------------------+             |
+---------------------------------------------------------------+
```

## Summary

| Slide | Keyword  | Question you answer         |
|-------|----------|-----------------------------|
| 1     | Setup    | "How did we keep it fair?"  |
| 2     | Numbers  | "What did we get?"          |
| 3     | Verdict  | "Did we beat the classics?" |

---

## Overall (one-slide / 30-second story)

**One sentence:** We feed historical OHLCV into engineered features, train an LSTM on past windows to predict next-bar direction, read off classification scores on the hold-out test, then run the same test period through a backtest to see how much money each strategy makes after fees.

```
  INPUT                    WHAT GOES IN              TRAIN                 OUT (MODEL SIDE)
  -----                    -------------             -----                 ----------------

  yfinance / CSV  ----->   OHLCV bars
  (per ticker,             + engineered features
   e.g. daily 5y           (RSI, MACD, BB width,
   or hourly 2y)           ATR, volume, patterns,
                            granularity flag)
                                    |
                                    v
                           Time-ordered split
                           train / val / test
                           (Scaler fit on TRAIN only)
                                    |
                                    v
                           LSTM (window -> logits)
                           optimize with BCE + pos_weight
                                    |
                                    v
                           Report on TEST (hold-out):
                           Accuracy, Precision,
                           Recall, F1  (threshold)
                                    |
                                    v
  BACKTEST                 Same TEST segment,
  --------                 same fee rule
                           LSTM signals vs
                           Buy-Hold / MACD / RSI / Bollinger
                                    |
                                    v
                           Money / risk:
                           total return, Sharpe,
                           max drawdown (+ equity curve)
```

**Talk track (short):**

1. **Input** — Downloaded market data: timestamped **open, high, low, close, volume**; we add **technical indicators** and a **granularity** flag so daily and hourly can share one pipeline.
2. **Train** — **Sliding windows** of features → LSTM predicts **next bar up/down**; we validate on val, report classification on **test**.
3. **Show metrics** — On **test**: **Accuracy, F1, Precision, Recall** (F1 matters when classes are imbalanced).
4. **Backtest** — Turn model **probabilities + threshold** into **long/short positions**; apply **transaction fee** on changes; compare to **four classical baselines** on the **same** test window.
5. **Money** — Read **total return**, **Sharpe**, **max drawdown** (and optionally plot **equity**) to answer “did it make money / beat baselines?”
