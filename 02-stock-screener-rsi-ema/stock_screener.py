# ================================
# IMPROVED STOCK SCREENER (RSI + EMA)
# ================================

import pandas as pd
import yfinance as yf
import pandas_ta as ta

# --- Tickers to Scan ---
TICKERS = [
    "NIFTYBEES.NS", "GOLDBEES.NS", "RELIANCE.NS", "TCS.NS", "INFY.NS",
    "HDFCBANK.NS", "ICICIBANK.NS", "TATAMOTORS.NS", "SBIN.NS",
    "KOTAKBANK.NS", "LT.NS", "ITC.NS"
]

PERIOD = "200d"
INTERVAL = "1d"

results = []

for ticker in TICKERS:
    print(f"Processing {ticker} ...")

    try:
        data = yf.download(
            ticker,
            period=PERIOD,
            interval=INTERVAL,
            progress=False,
            threads=True
        )

        if data.empty:
            print(f"  -> No data for {ticker}, skipping.\n")
            continue

        # Fix MultiIndex issue
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        data = data[["Open", "High", "Low", "Close", "Volume"]].dropna()

        # Indicators
        data["EMA_20"] = ta.ema(data["Close"], length=20)
        data["EMA_50"] = ta.ema(data["Close"], length=50)
        data["RSI_14"] = ta.rsi(data["Close"], length=14)

        data = data.dropna()

        last = data.iloc[-1]

        close = last["Close"]
        ema20 = last["EMA_20"]
        ema50 = last["EMA_50"]
        rsi14 = last["RSI_14"]

        # Screener conditions
        uptrend = ema20 > ema50
        price_above_ema20 = close > ema20

        # Momentum filter
        rsi_ok = (rsi14 > 50) & (rsi14 < 70)  # better momentum rule

        passes_screen = uptrend and price_above_ema20 and rsi_ok

        results.append({
            "Ticker": ticker,
            "Close": round(close, 2),
            "EMA_20": round(ema20, 2),
            "EMA_50": round(ema50, 2),
            "RSI_14": round(rsi14, 2),
            "Uptrend": uptrend,
            "Price > EMA20": price_above_ema20,
            "RSI Between 50–70": rsi_ok,
            "PASS_SCREEN": passes_screen
        })

    except Exception as e:
        print(f"  -> Error for {ticker}: {e}\n")
        continue

# Convert to DataFrame
results_df = pd.DataFrame(results)

print("\n===== FULL SCREENER OUTPUT =====\n")
print(results_df)

candidates = results_df[results_df["PASS_SCREEN"] == True].copy()

print("\n===== STOCKS PASSING THE SCREEN =====\n")
if candidates.empty:
    print("No stocks passed the screen.")
else:
    candidates = candidates.sort_values(by="RSI_14")
    print(candidates)
