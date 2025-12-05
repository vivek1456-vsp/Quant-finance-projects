# =======================================
# RSI + EMA TRADING STRATEGY (GOLDBEES)
# =======================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_ta as ta

# -------------------------------
# 1. Download data
# -------------------------------
TICKER = "GOLDBEES.NS"
START = "2024-01-01"
END   = "2025-12-04"

stock = yf.download(TICKER, start=START, end=END)

if stock.empty:
    raise SystemExit("No data downloaded. Check ticker or date range.")

# If yfinance returns MultiIndex columns (sometimes happens), flatten them
if isinstance(stock.columns, pd.MultiIndex):
    stock.columns = stock.columns.droplevel(1)

# Keep only needed columns
stock = stock[["Open", "High", "Low", "Close", "Volume"]].dropna()

# -------------------------------
# 2. Indicators (RSI + EMAs)
# -------------------------------
stock["RSI"] = ta.rsi(stock["Close"], length=14)
stock["EMA_20"] = ta.ema(stock["Close"], length=20)
stock["EMA_50"] = ta.ema(stock["Close"], length=50)

# Drop rows where indicators are not ready
stock = stock.dropna(subset=["RSI", "EMA_20", "EMA_50"]).copy()

if stock.empty:
    raise SystemExit("No data left after indicators. Try older START date.")

# -------------------------------
# 3. BUY SIGNAL (Trend + RSI filter)
# -------------------------------
# Buy when:
#  - EMA_20 > EMA_50 (uptrend)
#  - RSI > 55 (momentum confirmation)
stock["BUYSIGNAL"] = (stock["EMA_20"] > stock["EMA_50"]) & (stock["RSI"] > 55)

# -------------------------------
# 4. ENTRY / EXIT LOGIC
# -------------------------------
# ENTRY: BUYSIGNAL turns from False -> True
# EXIT : BUYSIGNAL turns from True  -> False
stock["ENTRY"] = (stock["BUYSIGNAL"].shift(1) == False) & (stock["BUYSIGNAL"] == True)
stock["EXIT"]  = (stock["BUYSIGNAL"].shift(1) == True) & (stock["BUYSIGNAL"] == False)

# -------------------------------
# 5. BUILD TRADES TABLE USING A LOOP
# -------------------------------
trades_list = []
in_trade = False
entry_price = None
entry_date = None

for date, row in stock.iterrows():
    if (not in_trade) and row["ENTRY"]:
        # Open trade
        in_trade = True
        entry_date = date
        entry_price = row["Close"]

    elif in_trade and row["EXIT"]:
        # Close trade
        exit_date = date
        exit_price = row["Close"]
        trade_return = (exit_price - entry_price) / entry_price * 100

        trades_list.append({
            "EntryDate": entry_date,
            "ExitDate": exit_date,
            "EntryPrice": entry_price,
            "ExitPrice": exit_price,
            "RETURN%": trade_return
        })

        in_trade = False
        entry_price = None
        entry_date = None

# If last trade is still open at the end, close it at last available close price
if in_trade:
    exit_date = stock.index[-1]
    exit_price = stock["Close"].iloc[-1]
    trade_return = (exit_price - entry_price) / entry_price * 100
    trades_list.append({
        "EntryDate": entry_date,
        "ExitDate": exit_date,
        "EntryPrice": entry_price,
        "ExitPrice": exit_price,
        "RETURN%": trade_return
    })

trades = pd.DataFrame(trades_list)

# -------------------------------
# 6. PRINT TRADES & STATS
# -------------------------------
print("\n===== TRADES =====\n")
if trades.empty:
    print("No trades were taken with this strategy.")
else:
    print(trades[["EntryDate", "ExitDate", "EntryPrice", "ExitPrice", "RETURN%"]])
    print(f"\nTotal Trades: {len(trades)}")
    print(f"Average Return per Trade: {trades['RETURN%'].mean():.2f}%")
    print(f"Total Return (sum of trades): {trades['RETURN%'].sum():.2f}%")

# -------------------------------
# 7. PLOTTING
# -------------------------------
plt.figure(figsize=(12, 8))

# (A) Price + EMAs + Entry/Exit points
plt.subplot(2, 1, 1)
plt.plot(stock.index, stock["Close"], label="Close", color="pink")
plt.plot(stock.index, stock["EMA_20"], label="EMA 20", color="green")
plt.plot(stock.index, stock["EMA_50"], label="EMA 50", color="red")

# plot entry + exit markers if trades exist
if not trades.empty:
    plt.scatter(trades["EntryDate"], trades["EntryPrice"], marker="^", color="green", s=80, label="Entry")
    plt.scatter(trades["ExitDate"], trades["ExitPrice"], marker="v", color="red", s=80, label="Exit")

plt.title(f"{TICKER} - Price with EMA20 & EMA50 and Trades")
plt.ylabel("Price")
plt.legend()
plt.grid(True)

# (B) RSI
plt.subplot(2, 1, 2)
plt.plot(stock.index, stock["RSI"], label="RSI (14)", color="blue")
plt.axhline(30, linestyle="--", color="green", alpha=0.7)
plt.axhline(70, linestyle="--", color="red", alpha=0.7)
plt.axhline(55, linestyle="--", color="purple", alpha=0.7, label="Buy Level 55")
plt.title("RSI Indicator")
plt.ylabel("RSI")
plt.xlabel("Date")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Optional: see first few rows in console
# print(stock.head())
