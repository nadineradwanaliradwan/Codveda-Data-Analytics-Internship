"""
Codveda Data Analytics Internship
Level 2 - Task 2: Time Series Analysis
Dataset: Stock Prices Data Set (AAPL focus)
Author: Nadine
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1. Load & Prepare Dataset
# ─────────────────────────────────────────
df_raw = pd.read_csv("datasets/Data Set For Task/2) Stock Prices Data Set.csv")
df_raw['date'] = pd.to_datetime(df_raw['date'])

print("=" * 55)
print("STEP 1: Dataset Overview")
print("=" * 55)
print(f"Shape: {df_raw.shape}")
print(f"Date range: {df_raw['date'].min()} → {df_raw['date'].max()}")
print(f"Symbols: {df_raw['symbol'].nunique()} stocks")
print(f"Missing values: {df_raw.isnull().sum().sum()}")

# Focus on AAPL for clear demonstration
df = df_raw[df_raw['symbol'] == 'AAPL'].copy()
df = df.sort_values('date').reset_index(drop=True)
df = df.dropna(subset=['close'])
df.set_index('date', inplace=True)

print(f"\nFocus: AAPL — {len(df)} trading days")
print(df[['open', 'high', 'low', 'close', 'volume']].head())

# ─────────────────────────────────────────
# 2. Moving Average Smoothing
# ─────────────────────────────────────────
df['MA_30']  = df['close'].rolling(window=30).mean()
df['MA_90']  = df['close'].rolling(window=90).mean()
df['MA_200'] = df['close'].rolling(window=200).mean()

# Daily returns
df['daily_return'] = df['close'].pct_change() * 100

print("\n" + "=" * 55)
print("STEP 2: Moving Average Summary")
print("=" * 55)
print(f"  30-day MA  (latest): ${df['MA_30'].iloc[-1]:.2f}")
print(f"  90-day MA  (latest): ${df['MA_90'].iloc[-1]:.2f}")
print(f"  200-day MA (latest): ${df['MA_200'].iloc[-1]:.2f}")

# ─────────────────────────────────────────
# 3. Trend Decomposition (manual — no statsmodels needed)
#    Using rolling stats to approximate trend & seasonality
# ─────────────────────────────────────────
# Monthly resampling for trend
monthly = df['close'].resample('ME').mean()
yearly  = df['close'].resample('YE').mean()

# Residuals = actual - trend (30-day MA)
df['residual'] = df['close'] - df['MA_30']

print("\n" + "=" * 55)
print("STEP 3: Yearly Trend (Annual Average Close Price)")
print("=" * 55)
for date, val in yearly.items():
    print(f"  {date.year}: ${val:.2f}")

# ─────────────────────────────────────────
# 4. Volatility Analysis
# ─────────────────────────────────────────
df['volatility_30d'] = df['daily_return'].rolling(30).std()

print("\n" + "=" * 55)
print("STEP 4: Volatility & Return Stats")
print("=" * 55)
print(f"  Mean daily return : {df['daily_return'].mean():.3f}%")
print(f"  Std daily return  : {df['daily_return'].std():.3f}%")
print(f"  Max single-day gain : +{df['daily_return'].max():.2f}%")
print(f"  Max single-day loss :  {df['daily_return'].min():.2f}%")

# ─────────────────────────────────────────
# 5. Plots
# ─────────────────────────────────────────
fig = plt.figure(figsize=(16, 14))
fig.patch.set_facecolor('#f8f9fa')

# Plot 1: Price + Moving Averages
ax1 = fig.add_subplot(3, 2, (1, 2))
ax1.set_facecolor('#0d1117')
ax1.plot(df.index, df['close'], color='#58a6ff', linewidth=0.8, alpha=0.8, label='AAPL Close')
ax1.plot(df.index, df['MA_30'],  color='#f0883e', linewidth=1.5, label='30-Day MA')
ax1.plot(df.index, df['MA_90'],  color='#3fb950', linewidth=1.5, label='90-Day MA')
ax1.plot(df.index, df['MA_200'], color='#ff7b72', linewidth=2.0, label='200-Day MA')
ax1.set_title('AAPL Closing Price with Moving Averages', color='white', fontsize=13, fontweight='bold')
ax1.set_ylabel('Price ($)', color='white')
ax1.tick_params(colors='white')
ax1.legend(facecolor='#161b22', labelcolor='white')
ax1.grid(alpha=0.15, color='white')
for spine in ax1.spines.values():
    spine.set_edgecolor('#30363d')

# Plot 2: Yearly trend bar chart
ax2 = fig.add_subplot(3, 2, 3)
ax2.set_facecolor('#f0f0f0')
years = [d.year for d in yearly.index]
vals  = yearly.values
colors_bar = ['#4C72B0' if v < yearly.values.max() else '#C44E52' for v in vals]
bars = ax2.bar(years, vals, color=colors_bar, edgecolor='white', width=0.6)
for bar, val in zip(bars, vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'${val:.0f}', ha='center', fontsize=7, fontweight='bold')
ax2.set_title('Yearly Average Close Price (Trend)', fontweight='bold')
ax2.set_ylabel('Avg Price ($)')
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Daily Returns Distribution
ax3 = fig.add_subplot(3, 2, 4)
ax3.set_facecolor('#f0f0f0')
returns = df['daily_return'].dropna()
ax3.hist(returns, bins=80, color='#4C72B0', edgecolor='white', alpha=0.8)
ax3.axvline(returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={returns.mean():.2f}%')
ax3.axvline(0, color='black', linestyle='-', linewidth=1)
ax3.set_title('Daily Returns Distribution', fontweight='bold')
ax3.set_xlabel('Daily Return (%)')
ax3.set_ylabel('Frequency')
ax3.legend()
ax3.grid(alpha=0.3)

# Plot 4: Residuals (actual - trend)
ax4 = fig.add_subplot(3, 2, 5)
ax4.set_facecolor('#f0f0f0')
ax4.plot(df.index, df['residual'], color='#9B59B6', linewidth=0.7, alpha=0.8)
ax4.axhline(0, color='red', linestyle='--', linewidth=1.5)
ax4.fill_between(df.index, df['residual'], 0,
                 where=df['residual'] > 0, color='#2ecc71', alpha=0.3, label='Above trend')
ax4.fill_between(df.index, df['residual'], 0,
                 where=df['residual'] < 0, color='#e74c3c', alpha=0.3, label='Below trend')
ax4.set_title('Residuals (Price − 30d MA)', fontweight='bold')
ax4.set_ylabel('Residual ($)')
ax4.legend()
ax4.grid(alpha=0.3)

# Plot 5: Rolling Volatility
ax5 = fig.add_subplot(3, 2, 6)
ax5.set_facecolor('#f0f0f0')
ax5.plot(df.index, df['volatility_30d'], color='#e67e22', linewidth=1.0)
ax5.fill_between(df.index, df['volatility_30d'], alpha=0.3, color='#e67e22')
ax5.set_title('30-Day Rolling Volatility', fontweight='bold')
ax5.set_ylabel('Volatility (Std of Daily Return %)')
ax5.grid(alpha=0.3)

plt.suptitle('AAPL Stock Price – Time Series Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('L2_T2_TimeSeries.png', dpi=150, bbox_inches='tight')
print("\n✅ Time series plots saved to: L2_T2_TimeSeries.png")

print("\n" + "=" * 55)
print("KEY INSIGHTS")
print("=" * 55)
print("• AAPL shows a strong long-term upward trend across all years")
print("• 200-day MA confirms sustained bullish momentum")
print("• Daily returns approximately normally distributed (efficient market)")
print("• Volatility spikes visible during market stress periods")
print("• Residuals show short-term oscillations around the trend")
