import os, argparse, datetime, sys
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("未安裝 yfinance，請先執行: pip install yfinance")
    sys.exit(1)

SAVE_DIR = "C:/Users/ryanb/Desktop/work/python"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- 解析參數 ----------
def get_args():
    p = argparse.ArgumentParser("Stock OHLCV Downloader (interactive-friendly)")
    p.add_argument("--ticker", nargs="+", help="股票代碼 (空格可多檔)")
    p.add_argument("--years", type=float, help="回溯年數 (float，可 0.x)")
    p.add_argument("--interval", type=str, help="資料間隔 1m/1h/1d/1wk…")
    a = p.parse_args()

    if not a.ticker:
        t = input("請輸入股票代碼: ").strip()
        if not t:
            print("⚠️  未輸入股票代碼。")
            sys.exit(0)
        a.ticker = t.split()

    if a.years is None:
        y = input("請輸入回溯年數 (預設 5): ").strip()
        a.years = float(y) if y.replace(".", "", 1).isdigit() and float(y) > 0 else 5

    if a.interval is None:
        it = input("請輸入資料間隔 (預設 1d): ").strip()
        a.interval = it or "1d"

    return a

args = get_args()

# ---------- interval 支援範圍 ----------
interval_limits = {
    "1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60,
    "60m": 730, "90m": 730, "1h": 730,
    "1d": 3650, "5d": 3650, "1wk": 3650, "1mo": 3650, "3mo": 3650
}
if args.interval not in interval_limits:
    print(f"❌ 不支援的 interval「{args.interval}」。可用：{', '.join(interval_limits)}")
    sys.exit(1)

# ---------- 自動限制回溯天數 ----------
max_days = interval_limits[args.interval]
max_years = max_days / 365
if args.years * 365 > max_days:
    print(f"⚠️  {args.interval} 最多支援 {max_days} 天，已調整為 {max_years:.3f} 年")
    args.years = max_years

if args.years <= 0:
    args.years = (1 / 365) if args.interval.endswith("m") or args.interval == "1h" else 1
    print("⚠️  回溯年數太小，已自動設為最小值")

start_date = datetime.date.today() - datetime.timedelta(days=int(args.years * 365))
end_date = datetime.date.today()

# ---------- 主迴圈 ----------
for tic in args.ticker:
    print(f"\n⏬ 下載 {tic} 近 {args.years:.3f} 年，每 {args.interval} 一筆…")
    try:
        df = yf.download(
            tic,
            start=start_date,
            end=end_date,
            interval=args.interval,
            auto_adjust=False,
            progress=False
        )
    except Exception as e:
        print(f"❌ {tic} 下載失敗: {e}")
        continue

    if df.empty:
        print(f"⚠️  {tic} 無資料，跳過。")
        continue

    # ---------- reset 並攤平欄位 ----------
    df = df.reset_index(drop=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(filter(None, map(str, col))).lower() for col in df.columns]
    else:
        df.columns = [str(col).lower() for col in df.columns]

    # ---------- 時間欄位偵測 ----------
    if "datetime" in df.columns:
        df.rename(columns={"datetime": "date"}, inplace=True)

    if "date" not in df.columns:
        print("⚠️  未偵測到時間欄位，跳過。")
        print("🧪 DataFrame 欄位：", df.columns.tolist())
        print("🧬 欄位型別：\n", df.dtypes)
        print("🔍 前 5 筆資料：\n", df.head())
        continue

    # ---------- 收盤價與其他欄位統一命名 ----------
    for base in ["open", "high", "low", "close", "volume"]:
        match = next((c for c in df.columns if c.startswith(base)), None)
        if match:
            df.rename(columns={match: base}, inplace=True)

    # ---------- 欄位完整性檢查 ----------
    needed = ["date", "open", "high", "low", "close", "volume"]
    if not set(needed).issubset(df.columns):
        print(f"⚠️  欄位缺失 {set(needed) - set(df.columns)}，跳過。")
        continue
    df = df[needed]

    # ---------- 輸出 CSV ----------
    fname = f"{tic.upper()}_{round(args.years, 3)}y_{args.interval}.csv"
    outpath = os.path.join(SAVE_DIR, fname)
    df.to_csv(outpath, index=False)
    print(f"✅ 已存檔 {outpath}")
