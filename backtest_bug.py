"""
Commodity-Equity Statistical Arbitrage — Backtesting Framework
===============================================================
Implements professor feedback:
  1. Grid search over (signal_window, k_sigma) parameter combinations
  2. No stop-loss/gain during optimization phase
  3. Portfolio-level capital allocation across pairs (Kelly / equal / vol-weighted)

Validated pairs:
  - Copper (HG=F) → TE Connectivity (TEL)
  - Copper (HG=F) → Rockwell Automation (ROK)

Run:
    pip install yfinance statsmodels pandas numpy matplotlib seaborn scipy
    python backtest.py
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import product
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import json
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

PAIRS = [
    ("HG=F", "TEL", "Copper → TE Connectivity"),
    ("HG=F", "ROK", "Copper → Rockwell Automation"),
]

START_DATE  = "2015-01-01"
END_DATE    = "2025-01-01"

# ── Train/Test Split ──────────────────────────────────────────────────────────
# 70% training (parameter optimization), 30% testing (out-of-sample validation)
TRAIN_RATIO = 0.70

# ── Beta Estimation ───────────────────────────────────────────────────────────
BETA_WINDOW = 252           # Rolling window for OLS beta (trading days)

# ── Grid Search Parameter Space (Professor feedback #1) ──────────────────────
# signal_window: lookback period over which commodity & equity moves are measured
# k_sigma: entry threshold in standard deviations
#
# Professor guidance: threshold should relate to window size
#   short window → higher k_sigma (noisier signal, need stronger filter)
#   long window  → lower k_sigma  (smoother signal, lower threshold OK)
#
# These pairings are built into the grid below.

SIGNAL_WINDOWS = [5, 10, 15, 20, 30, 45, 60, 90]   # days
K_SIGMA_MAP    = {                                   # window → candidate k_sigma values
     5: [1.5, 2.0, 2.5],
    10: [1.5, 2.0, 2.5],
    15: [1.2, 1.5, 2.0],
    20: [1.2, 1.5, 2.0],
    30: [1.0, 1.2, 1.5],
    45: [0.8, 1.0, 1.2],
    60: [0.8, 1.0, 1.2],
    90: [0.6, 0.8, 1.0],
}

ZSCORE_WINDOW  = 60          # Rolling window for Z-score normalization
EXIT_THRESHOLD = 0.25        # Z-score level at which position is closed (mean reversion)
MAX_HOLD_DAYS  = 20          # Maximum holding period before forced exit

# ── Capital Allocation (Professor feedback #3) ────────────────────────────────
TOTAL_CAPITAL      = 100_000   # USD
ALLOCATION_METHOD  = "vol_weighted"  # Options: "equal", "vol_weighted", "kelly"
MAX_POSITION_FRAC  = 0.40    # Max fraction of capital per pair (40%)
MIN_POSITION_FRAC  = 0.10    # Min fraction of capital per pair (10%)

# ── Transaction Costs ─────────────────────────────────────────────────────────
COMMISSION_PER_TRADE = 0.001   # 0.10% round-trip per trade


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATA COLLECTION
# ═══════════════════════════════════════════════════════════════════════════════

def download_pair(commodity_ticker, equity_ticker, start, end):
    """Downloads and aligns daily closing prices for a commodity-equity pair."""
    print(f"  Downloading {commodity_ticker} & {equity_ticker}...")

    c = yf.download(commodity_ticker, start=start, end=end,
                    progress=False, auto_adjust=True)["Close"]
    e = yf.download(equity_ticker,   start=start, end=end,
                    progress=False, auto_adjust=True)["Close"]

    if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
    if isinstance(e, pd.DataFrame): e = e.iloc[:, 0]

    df = pd.DataFrame({"commodity": c, "equity": e}).ffill().dropna()
    print(f"  {len(df)} trading days  ({df.index[0].date()} → {df.index[-1].date()})")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SIGNAL CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rolling_beta(df, beta_window=BETA_WINDOW):
    """
    Computes rolling OLS beta using weekly returns for stability.
    Beta tells us how much the equity should move per unit of commodity move.
    """
    weekly = df.resample("W").last().pct_change().dropna()

    rolling_beta = (
        weekly["equity"].rolling(beta_window // 5)
        .cov(weekly["commodity"])
        / weekly["commodity"].rolling(beta_window // 5).var()
    )

    # Reindex back to daily, forward-fill
    daily_beta = rolling_beta.reindex(df.index, method="ffill")
    return daily_beta


def compute_spread_zscore(df, beta_series, signal_window, zscore_window=ZSCORE_WINDOW):
    """
    Computes the mispricing spread and its rolling Z-score.

    Spread(t) = (beta × commodity_return_over_window) - equity_return_over_window

    Positive spread: equity underreacted to commodity RISE  → short equity
    Negative spread: equity underreacted to commodity FALL  → long equity
    """
    commodity_ret = df["commodity"].pct_change(signal_window)
    equity_ret    = df["equity"].pct_change(signal_window)

    spread = (beta_series * commodity_ret) - equity_ret

    roll_mean = spread.rolling(zscore_window).mean()
    roll_std  = spread.rolling(zscore_window).std()
    zscore    = (spread - roll_mean) / roll_std

    return spread, zscore


# ═══════════════════════════════════════════════════════════════════════════════
# 4. BACKTESTING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest(df, beta_series, signal_window, k_sigma,
                 capital=10_000, commission=COMMISSION_PER_TRADE,
                 exit_threshold=EXIT_THRESHOLD, max_hold=MAX_HOLD_DAYS):
    """
    Simulates the trading strategy on a single pair.

    Entry:  |Z-score| > k_sigma
    Exit:   |Z-score| < exit_threshold  OR  max_hold days elapsed
    No stop-loss (Professor feedback #2 — add later after optimization)

    Returns a DataFrame of trades and a portfolio equity curve.
    """
    spread, zscore = compute_spread_zscore(df, beta_series, signal_window)

    equity_prices = df["equity"]
    dates         = df.index

    # State variables
    position       = 0       # +1 = long equity, -1 = short equity, 0 = flat
    entry_price    = 0.0
    entry_date     = None
    hold_days      = 0
    cash           = capital
    shares         = 0.0
    portfolio_val  = capital

    trades         = []
    equity_curve   = []

    for i in range(len(dates)):
        date  = dates[i]
        price = float(equity_prices.iloc[i])
        z     = float(zscore.iloc[i]) if not np.isnan(zscore.iloc[i]) else 0.0

        # ── Mark-to-market portfolio value ──
        if position != 0:
            if position == 1:    # Long
                portfolio_val = cash + shares * price
            else:                # Short
                portfolio_val = cash + shares * (entry_price - price + entry_price)
                # Simplified short P&L: profit = entry - current price

        equity_curve.append({"date": date, "portfolio_value": portfolio_val,
                              "zscore": z, "position": position})

        # ── Exit logic ──
        if position != 0:
            hold_days += 1
            revert  = abs(z) < exit_threshold
            timeout = hold_days >= max_hold

            if revert or timeout:
                exit_price = price
                if position == 1:
                    pnl = (exit_price - entry_price) * shares
                else:
                    pnl = (entry_price - exit_price) * shares

                cost = commission * abs(shares) * exit_price
                pnl -= cost
                cash += shares * exit_price if position == 1 else \
                        shares * (2 * entry_price - exit_price)
                cash -= cost

                trades.append({
                    "entry_date":  entry_date,
                    "exit_date":   date,
                    "direction":   "LONG" if position == 1 else "SHORT",
                    "entry_price": round(entry_price, 4),
                    "exit_price":  round(exit_price, 4),
                    "hold_days":   hold_days,
                    "pnl":         round(pnl, 2),
                    "exit_reason": "revert" if revert else "timeout",
                })

                position  = 0
                shares    = 0.0
                hold_days = 0

        # ── Entry logic ──
        if position == 0 and not np.isnan(z) and abs(z) > k_sigma:
            trade_capital = cash
            shares        = trade_capital / price
            entry_price   = price
            entry_date    = date
            hold_days     = 0

            cost = commission * shares * price
            cash -= cost

            if z > k_sigma:        # Commodity rose, equity hasn't fallen yet
                position = -1      # Short equity
                shares   = trade_capital / price
            else:                  # Commodity fell, equity hasn't risen yet
                position = 1       # Long equity
                shares   = trade_capital / price

    ec_df     = pd.DataFrame(equity_curve).set_index("date")
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    return ec_df, trades_df


def compute_metrics(equity_curve, trades_df, capital):
    """
    Computes key performance metrics from backtest results.
    """
    if equity_curve.empty or trades_df.empty:
        return {}

    pv = equity_curve["portfolio_value"]

    # Returns
    total_return  = (pv.iloc[-1] - capital) / capital * 100
    daily_returns = pv.pct_change().dropna()

    # Sharpe ratio (annualized, risk-free = 0 for simplicity)
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)
              if daily_returns.std() > 0 else 0)

    # Maximum drawdown
    rolling_max = pv.cummax()
    drawdown     = (pv - rolling_max) / rolling_max
    max_dd       = drawdown.min() * 100

    # Trade stats
    wins      = trades_df[trades_df["pnl"] > 0]
    losses    = trades_df[trades_df["pnl"] <= 0]
    win_rate  = len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0
    avg_win   = wins["pnl"].mean()   if len(wins)   > 0 else 0
    avg_loss  = losses["pnl"].mean() if len(losses) > 0 else 0
    profit_factor = (wins["pnl"].sum() / abs(losses["pnl"].sum())
                     if losses["pnl"].sum() != 0 else np.inf)

    return {
        "total_return_pct":  round(total_return, 2),
        "sharpe_ratio":      round(sharpe, 3),
        "max_drawdown_pct":  round(max_dd, 2),
        "num_trades":        len(trades_df),
        "win_rate_pct":      round(win_rate, 2),
        "avg_win":           round(avg_win, 2),
        "avg_loss":          round(avg_loss, 2),
        "profit_factor":     round(profit_factor, 3),
        "avg_hold_days":     round(trades_df["hold_days"].mean(), 1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. GRID SEARCH (Professor Feedback #1)
# ═══════════════════════════════════════════════════════════════════════════════

def grid_search(df, beta_series, label, train_end_idx):
    """
    Exhaustively tests all (signal_window, k_sigma) combinations on the
    training set. Returns a sorted results DataFrame.

    Professor's insight: k_sigma should scale inversely with window length.
    Short windows are noisy → need higher threshold to filter noise.
    Long windows are smoother → lower threshold captures real signals.
    """
    print(f"\n  Grid search on TRAINING data ({label})...")

    train_df   = df.iloc[:train_end_idx]
    train_beta = beta_series.iloc[:train_end_idx]

    results = []

    all_combos = []
    for sw in SIGNAL_WINDOWS:
        for k in K_SIGMA_MAP[sw]:
            all_combos.append((sw, k))

    for sw, k in all_combos:
        try:
            ec, trades = run_backtest(
                train_df, train_beta,
                signal_window=sw, k_sigma=k,
                capital=10_000
            )
            metrics = compute_metrics(ec, trades, 10_000)
            if metrics:
                results.append({
                    "signal_window": sw,
                    "k_sigma":       k,
                    **metrics
                })
        except Exception as e:
            pass   # Skip invalid combinations silently

    results_df = pd.DataFrame(results)
    if results_df.empty:
        return results_df

    # Sort by Sharpe ratio (primary), then total return (secondary)
    results_df = results_df.sort_values(
        ["sharpe_ratio", "total_return_pct"], ascending=[False, False]
    ).reset_index(drop=True)

    print(f"  Tested {len(results_df)} parameter combinations")
    print(f"  Best: window={results_df.iloc[0]['signal_window']}d  "
          f"k={results_df.iloc[0]['k_sigma']}σ  "
          f"Sharpe={results_df.iloc[0]['sharpe_ratio']}  "
          f"Return={results_df.iloc[0]['total_return_pct']}%")

    return results_df


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CAPITAL ALLOCATION (Professor Feedback #3)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_allocations(pair_metrics, total_capital, method="vol_weighted"):
    """
    Allocates capital across pairs using one of three methods.

    equal:        Equal weight — simple, no bias
    vol_weighted: Inverse volatility — pairs with lower return volatility
                  receive more capital (risk parity approach)
    kelly:        Kelly criterion — size proportional to edge / variance,
                  capped to prevent over-concentration
    """
    n = len(pair_metrics)
    if n == 0:
        return {}

    labels = list(pair_metrics.keys())

    if method == "equal":
        weights = {label: 1/n for label in labels}

    elif method == "vol_weighted":
        # Inverse of max drawdown as volatility proxy (larger drawdown = lower allocation)
        raw    = {l: 1 / (abs(m["max_drawdown_pct"]) + 1e-6)
                  for l, m in pair_metrics.items()}
        total  = sum(raw.values())
        weights = {l: v/total for l, v in raw.items()}

    elif method == "kelly":
        # Simplified Kelly: f = (win_rate/avg_loss - loss_rate/avg_win) clipped
        weights_raw = {}
        for label, m in pair_metrics.items():
            wr   = m["win_rate_pct"] / 100
            lr   = 1 - wr
            aw   = abs(m["avg_win"])   + 1e-6
            al   = abs(m["avg_loss"])  + 1e-6
            kelly = (wr / al) - (lr / aw)
            weights_raw[label] = max(kelly, 0.01)   # Floor at 1%
        total = sum(weights_raw.values())
        weights = {l: v/total for l, v in weights_raw.items()}

    # Apply min/max caps and renormalize
    weights = {l: np.clip(w, MIN_POSITION_FRAC, MAX_POSITION_FRAC)
               for l, w in weights.items()}
    total   = sum(weights.values())
    weights = {l: w/total for l, w in weights.items()}

    allocations = {l: weights[l] * total_capital for l in labels}
    return allocations


# ═══════════════════════════════════════════════════════════════════════════════
# 7. VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_grid_search_heatmap(grid_df, label, save_path):
    """Heatmap of Sharpe ratio across all (signal_window, k_sigma) combinations."""
    if grid_df.empty:
        return

    pivot = grid_df.pivot_table(
        index="k_sigma", columns="signal_window", values="sharpe_ratio"
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=pivot.values.min(), vmax=pivot.values.max())

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c}d" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r}σ" for r in pivot.index])
    ax.set_xlabel("Signal Window (days)", fontsize=11)
    ax.set_ylabel("Entry Threshold (k_sigma)", fontsize=11)
    ax.set_title(f"Grid Search — Sharpe Ratio Heatmap\n{label}", fontsize=12, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Sharpe Ratio")

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if 0.3 < val < 1.5 else "white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Heatmap saved → {save_path}")


def plot_equity_curves(train_ec, test_ec, label, best_params, save_path):
    """Plots train and test equity curves with Z-score overlay."""
    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 1, hspace=0.4)

    colors = {"train": "#2E75B6", "test": "#2E8B57"}

    # ── Panel 1: Portfolio Value ──
    ax1 = fig.add_subplot(gs[0])
    if not train_ec.empty:
        ax1.plot(train_ec.index, train_ec["portfolio_value"],
                 color=colors["train"], linewidth=1.2, label="Training period")
    if not test_ec.empty:
        ax1.plot(test_ec.index, test_ec["portfolio_value"],
                 color=colors["test"], linewidth=1.2, label="Test period (OOS)")
    if not train_ec.empty and not test_ec.empty:
        split_date = test_ec.index[0]
        ax1.axvline(split_date, color="red", linestyle="--",
                    linewidth=1.0, label="Train/Test split")
    ax1.set_title(f"Portfolio Equity Curve — {label}\n"
                  f"Optimized params: window={best_params['signal_window']}d  "
                  f"k={best_params['k_sigma']}σ", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # ── Panel 2: Z-score ──
    ax2 = fig.add_subplot(gs[1])
    all_ec = pd.concat([train_ec, test_ec]) if not train_ec.empty else test_ec
    ax2.plot(all_ec.index, all_ec["zscore"],
             color="#5B5EA6", linewidth=0.8, alpha=0.7, label="Z-score")
    ax2.axhline( best_params["k_sigma"],  color="#E07B39", linestyle="--",
                linewidth=1.0, label=f"+{best_params['k_sigma']}σ Short entry")
    ax2.axhline(-best_params["k_sigma"],  color="#3A7EBF", linestyle="--",
                linewidth=1.0, label=f"-{best_params['k_sigma']}σ Long entry")
    ax2.axhline( EXIT_THRESHOLD, color="green",  linestyle=":", linewidth=0.8)
    ax2.axhline(-EXIT_THRESHOLD, color="green",  linestyle=":", linewidth=0.8, label="Exit ±0.25σ")
    ax2.axhline(0, color="black", linestyle="-", linewidth=0.5)
    if not train_ec.empty and not test_ec.empty:
        ax2.axvline(split_date, color="red", linestyle="--", linewidth=1.0)
    ax2.set_title("Z-Score Signal Over Full Period", fontsize=11)
    ax2.set_ylabel("Z-Score")
    ax2.legend(fontsize=8, ncol=3)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(-5, 5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Equity curve saved → {save_path}")


def plot_allocation_chart(allocations, save_path):
    """Bar chart of capital allocation across pairs."""
    fig, ax = plt.subplots(figsize=(8, 4))

    labels = [l.replace(" → ", "\n→ ") for l in allocations.keys()]
    values = list(allocations.values())
    colors = ["#2E75B6", "#2E8B57", "#E07B39", "#9B59B6"][:len(labels)]

    bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="white")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f"${val:,.0f}\n({val/sum(values)*100:.1f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title(f"Capital Allocation ({ALLOCATION_METHOD.replace('_',' ').title()} Method)\n"
                 f"Total Capital: ${sum(values):,.0f}", fontsize=12, fontweight="bold")
    ax.set_ylabel("Allocated Capital ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Allocation chart saved → {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline():
    print("=" * 65)
    print("COMMODITY-EQUITY ARBITRAGE — BACKTESTING PIPELINE")
    print(f"Period: {START_DATE} → {END_DATE}  |  Train: {int(TRAIN_RATIO*100)}%  Test: {int((1-TRAIN_RATIO)*100)}%")
    print(f"Allocation method: {ALLOCATION_METHOD}")
    print("=" * 65)

    all_results  = {}   # Stores results per pair
    pair_metrics = {}   # Best-param metrics for allocation step

    for commodity_ticker, equity_ticker, label in PAIRS:
        print(f"\n{'─' * 65}")
        print(f"PAIR: {label}")
        print(f"{'─' * 65}")

        # ── Download data ──
        df = download_pair(commodity_ticker, equity_ticker, START_DATE, END_DATE)

        # ── Train/test split ──
        split_idx  = int(len(df) * TRAIN_RATIO)
        split_date = df.index[split_idx]
        train_df   = df.iloc[:split_idx]
        test_df    = df.iloc[split_idx:]
        print(f"  Train: {train_df.index[0].date()} → {train_df.index[-1].date()} ({len(train_df)} days)")
        print(f"  Test:  {test_df.index[0].date()}  → {test_df.index[-1].date()}  ({len(test_df)} days)")

        # ── Rolling beta ──
        beta_series = compute_rolling_beta(df)

        # ── Grid search on training data ──
        grid_df = grid_search(df, beta_series, label, split_idx)

        if grid_df.empty:
            print("  No valid results from grid search. Skipping pair.")
            continue

        # ── Extract best parameters ──
        best = grid_df.iloc[0]
        best_params = {
            "signal_window": int(best["signal_window"]),
            "k_sigma":       float(best["k_sigma"]),
        }
        print(f"\n  ── Best Parameters (Training) ──")
        print(f"  signal_window : {best_params['signal_window']} days")
        print(f"  k_sigma       : {best_params['k_sigma']}σ")
        print(f"  Sharpe        : {best['sharpe_ratio']}")
        print(f"  Total Return  : {best['total_return_pct']}%")
        print(f"  Win Rate      : {best['win_rate_pct']}%")
        print(f"  Max Drawdown  : {best['max_drawdown_pct']}%")
        print(f"  Num Trades    : {int(best['num_trades'])}")

        # ── Out-of-sample test with best parameters ──
        print(f"\n  ── Out-of-Sample Test ──")
        train_beta = beta_series.iloc[:split_idx]
        test_beta  = beta_series.iloc[split_idx:]

        train_ec, train_trades = run_backtest(
            train_df, train_beta, **best_params, capital=10_000)
        test_ec, test_trades = run_backtest(
            test_df, test_beta, **best_params, capital=10_000)

        train_metrics = compute_metrics(train_ec, train_trades, 10_000)
        test_metrics  = compute_metrics(test_ec,  test_trades,  10_000)

        print(f"  {'Metric':<25} {'Training':>12} {'Out-of-Sample':>15}")
        print(f"  {'─'*25} {'─'*12} {'─'*15}")
        for key in ["total_return_pct", "sharpe_ratio", "max_drawdown_pct",
                    "win_rate_pct", "num_trades", "avg_hold_days", "profit_factor"]:
            tv = train_metrics.get(key, "—")
            ov = test_metrics.get(key, "—")
            suffix = "%" if "pct" in key else ""
            print(f"  {key:<25} {str(tv)+suffix:>12} {str(ov)+suffix:>15}")

        # ── Store for allocation ──
        pair_metrics[label] = test_metrics   # Use OOS metrics for sizing

        # ── Save results ──
        safe = label.replace(" ", "_").replace("→", "to").replace("/", "-")

        plot_grid_search_heatmap(
            grid_df, label, f"grid_search_{safe}.png")

        plot_equity_curves(
            train_ec, test_ec, label, best_params,
            f"equity_curve_{safe}.png")

        all_results[label] = {
            "best_params":    best_params,
            "train_metrics":  train_metrics,
            "test_metrics":   test_metrics,
            "grid_top10":     grid_df.head(10).to_dict(orient="records"),
        }

        # Save top grid search results
        grid_df.head(20).to_csv(f"grid_results_{safe}.csv", index=False)
        print(f"  Grid results saved → grid_results_{safe}.csv")

    # ── Portfolio-level capital allocation ──
    print(f"\n{'─' * 65}")
    print(f"PORTFOLIO CAPITAL ALLOCATION ({ALLOCATION_METHOD.upper()})")
    print(f"{'─' * 65}")

    if pair_metrics:
        allocations = compute_allocations(
            pair_metrics, TOTAL_CAPITAL, method=ALLOCATION_METHOD)

        for label, alloc in allocations.items():
            pct = alloc / TOTAL_CAPITAL * 100
            print(f"  {label:<40} ${alloc:>10,.2f}  ({pct:.1f}%)")

        plot_allocation_chart(allocations, "capital_allocation.png")

        # Also show equal and kelly for comparison
        print(f"\n  Allocation comparison across methods:")
        print(f"  {'Pair':<40} {'Equal':>10} {'Vol-Wtd':>10} {'Kelly':>10}")
        print(f"  {'─'*40} {'─'*10} {'─'*10} {'─'*10}")
        for method in ["equal", "vol_weighted", "kelly"]:
            allocs = compute_allocations(pair_metrics, TOTAL_CAPITAL, method=method)
            if method == "equal":
                rows = {l: [f"${v:,.0f}"] for l, v in allocs.items()}
            elif method == "vol_weighted":
                for l, v in allocs.items(): rows[l].append(f"${v:,.0f}")
            else:
                for l, v in allocs.items(): rows[l].append(f"${v:,.0f}")

        for label, vals in rows.items():
            print(f"  {label:<40} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")

    # ── Save full results summary ──
    with open("backtest_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull results saved → backtest_results.json")

    # ── Final summary ──
    print(f"\n{'=' * 65}")
    print("FINAL SUMMARY")
    print(f"{'=' * 65}")
    for label, res in all_results.items():
        bp = res["best_params"]
        tm = res["test_metrics"]
        print(f"\n{label}")
        print(f"  Best params : window={bp['signal_window']}d, k={bp['k_sigma']}σ")
        print(f"  OOS Sharpe  : {tm.get('sharpe_ratio', '—')}")
        print(f"  OOS Return  : {tm.get('total_return_pct', '—')}%")
        print(f"  OOS Trades  : {tm.get('num_trades', '—')}")
        print(f"  OOS Win Rate: {tm.get('win_rate_pct', '—')}%")

    print(f"\nOutputs generated:")
    print(f"  grid_search_*.png        — Parameter heatmaps")
    print(f"  equity_curve_*.png       — Train/test equity curves")
    print(f"  capital_allocation.png   — Allocation bar chart")
    print(f"  grid_results_*.csv       — Top 20 parameter combinations per pair")
    print(f"  backtest_results.json    — Full structured results")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_pipeline()