"""
Commodity-Equity Statistical Arbitrage — FIXED Backtesting Framework
=====================================================================
FIXES:
  - Position sizing bug: now uses FIXED fraction of initial capital per trade
    (not full portfolio value), preventing exponential compounding explosion
  - Added all professor-required metrics:
    Sharpe, Calmar, Max Drawdown, Annualized Return (compounding + simple),
    Trading Frequency, Average Holding Period, Win Rate
  - Added scenario periods: 2008 crisis, COVID, US-China trade war
  - Added Back Test 2: nearest 6 months (mid-2024 to end-2024)

Run:
    python backtest_fixed.py
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import json

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

PAIRS = [
    ("HG=F", "TEL", "Copper → TE Connectivity"),
    ("HG=F", "ROK", "Copper → Rockwell Automation"),
]

# ── Data periods ──────────────────────────────────────────────────────────────
FULL_START  = "2012-01-01"   # Extended back for 2008-adjacent scenario testing
FULL_END    = "2025-01-01"

# ── Back Test 1: Full period with bull + bear conditions ──────────────────────
BT1_START   = "2015-01-01"
BT1_END     = "2025-01-01"
TRAIN_RATIO = 0.70           # 70% train / 30% OOS

# ── Back Test 2: Nearest 6 months ─────────────────────────────────────────────
BT2_START   = "2024-07-01"
BT2_END     = "2025-01-01"

# ── Scenario periods ──────────────────────────────────────────────────────────
SCENARIOS = {
    "2008 Financial Crisis":    ("2007-09-01", "2009-06-30"),
    "US-China Trade War":       ("2018-01-01", "2019-12-31"),
    "COVID Crash & Recovery":   ("2020-01-01", "2021-06-30"),
    "2022 Rate Hike Cycle":     ("2022-01-01", "2022-12-31"),
}

# ── Best parameters from grid search (use these for scenario + BT2 tests) ────
# TEL: window=90, k=0.6  |  ROK: window=45, k=0.8
BEST_PARAMS = {
    "Copper → TE Connectivity":    {"signal_window": 90, "k_sigma": 0.6},
    "Copper → Rockwell Automation":{"signal_window": 45, "k_sigma": 0.8},
}

# ── Position sizing (THE KEY FIX) ─────────────────────────────────────────────
POSITION_FRACTION = 0.95     # Fraction of capital deployed per trade
                             # Using ~95% leaves small buffer for commissions

BETA_WINDOW        = 252
ZSCORE_WINDOW      = 60
EXIT_THRESHOLD     = 0.25
MAX_HOLD_DAYS      = 20
COMMISSION         = 0.001   # 0.1% per trade
TOTAL_CAPITAL      = 100_000
RISK_FREE_RATE     = 0.04    # Annual, for Sharpe calculation

# ── Grid search space ─────────────────────────────────────────────────────────
SIGNAL_WINDOWS = [5, 10, 15, 20, 30, 45, 60, 90]
K_SIGMA_MAP    = {
     5: [1.5, 2.0, 2.5],
    10: [1.5, 2.0, 2.5],
    15: [1.2, 1.5, 2.0],
    20: [1.2, 1.5, 2.0],
    30: [1.0, 1.2, 1.5],
    45: [0.8, 1.0, 1.2],
    60: [0.8, 1.0, 1.2],
    90: [0.6, 0.8, 1.0],
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATA & SIGNAL
# ═══════════════════════════════════════════════════════════════════════════════

def download_pair(commodity_ticker, equity_ticker, start, end):
    c = yf.download(commodity_ticker, start=start, end=end,
                    progress=False, auto_adjust=True)["Close"]
    e = yf.download(equity_ticker,   start=start, end=end,
                    progress=False, auto_adjust=True)["Close"]
    if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
    if isinstance(e, pd.DataFrame): e = e.iloc[:, 0]
    return pd.DataFrame({"commodity": c, "equity": e}).ffill().dropna()


def compute_rolling_beta(df, beta_window=BETA_WINDOW):
    weekly     = df.resample("W").last().pct_change().dropna()
    roll_cov   = weekly["equity"].rolling(beta_window // 5).cov(weekly["commodity"])
    roll_var   = weekly["commodity"].rolling(beta_window // 5).var()
    roll_beta  = roll_cov / roll_var
    return roll_beta.reindex(df.index, method="ffill")


def compute_zscore(df, beta_series, signal_window, zscore_window=ZSCORE_WINDOW):
    c_ret  = df["commodity"].pct_change(signal_window)
    e_ret  = df["equity"].pct_change(signal_window)
    spread = (beta_series * c_ret) - e_ret
    zmean  = spread.rolling(zscore_window).mean()
    zstd   = spread.rolling(zscore_window).std()
    return spread, (spread - zmean) / zstd


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BACKTEST ENGINE — FIXED POSITION SIZING
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest(df, beta_series, signal_window, k_sigma,
                 capital=10_000,
                 commission=COMMISSION,
                 exit_threshold=EXIT_THRESHOLD,
                 max_hold=MAX_HOLD_DAYS,
                 position_fraction=POSITION_FRACTION):
    """
    KEY FIX: Each trade uses `position_fraction` of the INITIAL capital,
    not the current portfolio value. This prevents exponential compounding
    and gives realistic, interpretable returns.

    Trade P&L is accumulated additively into cash, which is the correct
    approach for a strategy that runs sequential trades from a fixed pool.
    """
    spread, zscore = compute_zscore(df, beta_series, signal_window)
    prices         = df["equity"]
    dates          = df.index

    trade_size = capital * position_fraction   # FIXED dollar amount per trade

    position    = 0
    entry_price = 0.0
    entry_date  = None
    hold_days   = 0
    cash        = capital

    trades       = []
    equity_curve = []

    for i in range(len(dates)):
        date  = dates[i]
        price = float(prices.iloc[i])
        z     = float(zscore.iloc[i]) if not np.isnan(zscore.iloc[i]) else 0.0

        # ── Mark-to-market ──
        if position != 0:
            shares_held = trade_size / entry_price
            if position == 1:    # Long: profit when price rises
                unrealized = (price - entry_price) * shares_held
            else:                # Short: profit when price falls
                unrealized = (entry_price - price) * shares_held
            portfolio_val = cash + unrealized
        else:
            portfolio_val = cash

        equity_curve.append({
            "date":            date,
            "portfolio_value": portfolio_val,
            "zscore":          z,
            "position":        position
        })

        # ── Exit logic ──
        if position != 0:
            hold_days += 1
            revert    = abs(z) < exit_threshold
            timeout   = hold_days >= max_hold

            if revert or timeout:
                shares = trade_size / entry_price
                if position == 1:
                    pnl = (price - entry_price) * shares
                else:
                    pnl = (entry_price - price) * shares

                cost  = commission * trade_size
                pnl  -= cost
                cash += pnl

                trades.append({
                    "entry_date":  entry_date,
                    "exit_date":   date,
                    "direction":   "LONG" if position == 1 else "SHORT",
                    "entry_price": round(entry_price, 4),
                    "exit_price":  round(price, 4),
                    "hold_days":   hold_days,
                    "pnl":         round(pnl, 4),
                    "pnl_pct":     round(pnl / trade_size * 100, 4),
                    "exit_reason": "revert" if revert else "timeout",
                })

                position  = 0
                hold_days = 0

        # ── Entry logic ──
        if position == 0 and not np.isnan(z) and abs(z) > k_sigma:
            entry_price = price
            entry_date  = date
            hold_days   = 0
            cost        = commission * trade_size
            cash       -= cost

            if z > k_sigma:
                position = -1   # Short: commodity rose, equity hasn't fallen
            else:
                position = 1    # Long:  commodity fell, equity hasn't risen

    ec_df     = pd.DataFrame(equity_curve).set_index("date")
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return ec_df, trades_df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. METRICS — ALL PROFESSOR-REQUIRED MEASURES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(equity_curve, trades_df, capital, label=""):
    """
    Computes all metrics specified by professor:
    Sharpe, Calmar, Max Drawdown, Expected Return (annualized compounding
    and simple non-compounding), Trading Frequency, Average Holding Period,
    Win Rate, Profit Factor.
    """
    if equity_curve.empty or trades_df is None or trades_df.empty:
        return {k: "—" for k in [
            "annualized_return_compounding", "annualized_return_simple",
            "sharpe_ratio", "calmar_ratio", "max_drawdown_pct",
            "trading_frequency_per_year", "avg_holding_period_days",
            "win_rate_pct", "profit_factor", "num_trades",
            "total_return_pct", "avg_win_pct", "avg_loss_pct"
        ]}

    pv            = equity_curve["portfolio_value"]
    daily_returns = pv.pct_change().dropna()
    n_days        = len(pv)
    n_years       = n_days / 252

    # ── Returns ──────────────────────────────────────────────────────────────
    total_return = (pv.iloc[-1] - capital) / capital

    # Compounding (CAGR): accounts for reinvestment effect
    cagr = (pv.iloc[-1] / capital) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Simple (arithmetic): total P&L / initial capital / years
    simple_annual = total_return / n_years if n_years > 0 else 0

    # ── Risk ──────────────────────────────────────────────────────────────────
    # Max Drawdown
    rolling_max  = pv.cummax()
    drawdown     = (pv - rolling_max) / rolling_max
    max_dd       = drawdown.min()

    # Sharpe (annualized, excess over risk-free)
    daily_rf     = RISK_FREE_RATE / 252
    excess_ret   = daily_returns - daily_rf
    sharpe       = (excess_ret.mean() / excess_ret.std() * np.sqrt(252)
                    if excess_ret.std() > 0 else 0)

    # Calmar = CAGR / |Max Drawdown| — measures return per unit of drawdown risk
    calmar       = cagr / abs(max_dd) if max_dd != 0 else 0

    # ── Trade stats ───────────────────────────────────────────────────────────
    wins         = trades_df[trades_df["pnl"] > 0]
    losses       = trades_df[trades_df["pnl"] <= 0]
    win_rate     = len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0
    avg_win_pct  = wins["pnl_pct"].mean()   if len(wins)   > 0 else 0
    avg_loss_pct = losses["pnl_pct"].mean() if len(losses) > 0 else 0

    profit_factor = (wins["pnl"].sum() / abs(losses["pnl"].sum())
                     if len(losses) > 0 and losses["pnl"].sum() != 0 else np.inf)

    # Trading frequency: trades per year
    trade_freq   = len(trades_df) / n_years if n_years > 0 else 0

    # Average holding period
    avg_hold     = trades_df["hold_days"].mean() if len(trades_df) > 0 else 0

    return {
        "total_return_pct":              round(total_return * 100, 2),
        "annualized_return_compounding": round(cagr * 100, 2),
        "annualized_return_simple":      round(simple_annual * 100, 2),
        "sharpe_ratio":                  round(sharpe, 3),
        "calmar_ratio":                  round(calmar, 3),
        "max_drawdown_pct":              round(max_dd * 100, 2),
        "trading_frequency_per_year":    round(trade_freq, 1),
        "avg_holding_period_days":       round(avg_hold, 1),
        "win_rate_pct":                  round(win_rate, 2),
        "profit_factor":                 round(profit_factor, 3),
        "num_trades":                    len(trades_df),
        "avg_win_pct":                   round(avg_win_pct, 3),
        "avg_loss_pct":                  round(avg_loss_pct, 3),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. GRID SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

def grid_search(df, beta_series, label, split_idx):
    print(f"  Grid search ({label})...")
    train_df   = df.iloc[:split_idx]
    train_beta = beta_series.iloc[:split_idx]
    results    = []

    for sw in SIGNAL_WINDOWS:
        for k in K_SIGMA_MAP[sw]:
            try:
                ec, trades = run_backtest(train_df, train_beta,
                                          signal_window=sw, k_sigma=k,
                                          capital=10_000)
                m = compute_metrics(ec, trades, 10_000)
                if m["sharpe_ratio"] != "—":
                    results.append({"signal_window": sw, "k_sigma": k, **m})
            except Exception:
                pass

    if not results:
        return pd.DataFrame()

    df_res = pd.DataFrame(results).sort_values(
        ["sharpe_ratio", "annualized_return_compounding"],
        ascending=[False, False]
    ).reset_index(drop=True)

    best = df_res.iloc[0]
    print(f"  Best → window={int(best['signal_window'])}d  "
          f"k={best['k_sigma']}σ  "
          f"Sharpe={best['sharpe_ratio']}  "
          f"CAGR={best['annualized_return_compounding']}%  "
          f"WinRate={best['win_rate_pct']}%")
    return df_res


# ═══════════════════════════════════════════════════════════════════════════════
# 6. VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_heatmap(grid_df, label, save_path):
    if grid_df.empty: return
    pivot = grid_df.pivot_table(
        index="k_sigma", columns="signal_window", values="sharpe_ratio")
    fig, ax = plt.subplots(figsize=(13, 5))
    vmax = min(pivot.values[~np.isnan(pivot.values)].max(), 5)
    vmin = max(pivot.values[~np.isnan(pivot.values)].min(), -1)
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c}d" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r}σ" for r in pivot.index])
    ax.set_xlabel("Signal Window (days)", fontsize=11)
    ax.set_ylabel("Entry Threshold (k_sigma)", fontsize=11)
    ax.set_title(f"Grid Search — Sharpe Ratio Heatmap\n{label}",
                 fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Sharpe Ratio")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if val > vmax * 0.7 or val < 0 else "black")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_equity_curves(train_ec, test_ec, label, best_params, save_path):
    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 1, hspace=0.4)

    ax1 = fig.add_subplot(gs[0])
    if not train_ec.empty:
        ax1.plot(train_ec.index, train_ec["portfolio_value"],
                 color="#2E75B6", linewidth=1.3, label="Training period")
    if not test_ec.empty:
        ax1.plot(test_ec.index, test_ec["portfolio_value"],
                 color="#2E8B57", linewidth=1.3, label="Test period (OOS)")
    if not train_ec.empty and not test_ec.empty:
        ax1.axvline(test_ec.index[0], color="red", linestyle="--",
                    linewidth=1.0, label="Train/Test split")
    ax1.set_title(
        f"Portfolio Equity Curve — {label}\n"
        f"Params: window={best_params['signal_window']}d  k={best_params['k_sigma']}σ  "
        f"| Position size: {POSITION_FRACTION*100:.0f}% of initial capital",
        fontsize=11, fontweight="bold")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    ax2 = fig.add_subplot(gs[1])
    all_ec = pd.concat([train_ec, test_ec]) if not train_ec.empty else test_ec
    ax2.plot(all_ec.index, all_ec["zscore"],
             color="#5B5EA6", linewidth=0.8, alpha=0.7)
    k = best_params["k_sigma"]
    ax2.axhline( k, color="#E07B39", linestyle="--", linewidth=1.0,
                label=f"+{k}σ Short")
    ax2.axhline(-k, color="#3A7EBF", linestyle="--", linewidth=1.0,
                label=f"-{k}σ Long")
    ax2.axhline( EXIT_THRESHOLD, color="green", linestyle=":", linewidth=0.8)
    ax2.axhline(-EXIT_THRESHOLD, color="green", linestyle=":", linewidth=0.8,
                label=f"Exit ±{EXIT_THRESHOLD}σ")
    ax2.axhline(0, color="black", linestyle="-", linewidth=0.5)
    if not train_ec.empty and not test_ec.empty:
        ax2.axvline(test_ec.index[0], color="red", linestyle="--", linewidth=1.0)
    ax2.set_title("Z-Score Signal", fontsize=11)
    ax2.set_ylabel("Z-Score")
    ax2.legend(fontsize=8, ncol=3)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(-5, 5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def print_metrics_table(metrics, label):
    print(f"\n  {'Metric':<40} {'Value':>12}")
    print(f"  {'─'*40} {'─'*12}")
    rows = [
        ("Total Return (%)",             "total_return_pct",              "%"),
        ("Ann. Return - Compounding (%)", "annualized_return_compounding", "%"),
        ("Ann. Return - Simple (%)",      "annualized_return_simple",      "%"),
        ("Sharpe Ratio",                  "sharpe_ratio",                  ""),
        ("Calmar Ratio",                  "calmar_ratio",                  ""),
        ("Max Drawdown (%)",              "max_drawdown_pct",              "%"),
        ("Trading Freq (trades/yr)",      "trading_frequency_per_year",    ""),
        ("Avg Holding Period (days)",     "avg_holding_period_days",       "d"),
        ("Win Rate (%)",                  "win_rate_pct",                  "%"),
        ("Profit Factor",                 "profit_factor",                 ""),
        ("Num Trades",                    "num_trades",                    ""),
        ("Avg Win (%)",                   "avg_win_pct",                   "%"),
        ("Avg Loss (%)",                  "avg_loss_pct",                  "%"),
    ]
    for display, key, suffix in rows:
        val = metrics.get(key, "—")
        print(f"  {display:<40} {str(val)+suffix:>12}")


def run_pipeline():
    print("=" * 65)
    print("FIXED BACKTESTING PIPELINE")
    print(f"Position size: {POSITION_FRACTION*100:.0f}% of initial capital (fixed)")
    print("=" * 65)

    all_output = {}

    for commodity_ticker, equity_ticker, label in PAIRS:
        print(f"\n{'═'*65}")
        print(f"PAIR: {label}")
        print(f"{'═'*65}")

        # ── Download full history ──────────────────────────────────────────────
        df_full = download_pair(commodity_ticker, equity_ticker,
                                FULL_START, FULL_END)
        df_bt1  = df_full[(df_full.index >= BT1_START) &
                          (df_full.index <  BT1_END)]
        df_bt2  = df_full[(df_full.index >= BT2_START) &
                          (df_full.index <  BT2_END)]

        beta_full = compute_rolling_beta(df_full)
        beta_bt1  = beta_full[df_bt1.index]
        beta_bt2  = beta_full[df_bt2.index]

        bp = BEST_PARAMS[label]

        # ── BACK TEST 1: Full period + train/test split ────────────────────────
        split_idx  = int(len(df_bt1) * TRAIN_RATIO)

        print(f"\n── BACK TEST 1 (Full Period: {BT1_START} → {BT1_END}) ──")
        grid_df = grid_search(df_bt1, beta_bt1, label, split_idx)

        if not grid_df.empty:
            best_row = grid_df.iloc[0]
            bp_grid  = {"signal_window": int(best_row["signal_window"]),
                        "k_sigma":       float(best_row["k_sigma"])}
        else:
            bp_grid = bp

        train_df, test_df     = df_bt1.iloc[:split_idx], df_bt1.iloc[split_idx:]
        train_b,  test_b      = beta_bt1.iloc[:split_idx], beta_bt1.iloc[split_idx:]

        train_ec, train_tr = run_backtest(train_df, train_b, capital=10_000, **bp_grid)
        test_ec,  test_tr  = run_backtest(test_df,  test_b,  capital=10_000, **bp_grid)

        train_m = compute_metrics(train_ec, train_tr, 10_000)
        test_m  = compute_metrics(test_ec,  test_tr,  10_000)

        print(f"\n  ── Training Set ({BT1_START} → split) ──")
        print_metrics_table(train_m, label)
        print(f"\n  ── Out-of-Sample Test ({int(TRAIN_RATIO*100)}%/{int((1-TRAIN_RATIO)*100)}% split) ──")
        print_metrics_table(test_m, label)

        safe = label.replace(" ","_").replace("→","to").replace("/","-")
        plot_equity_curves(train_ec, test_ec, label, bp_grid,
                           f"equity_fixed_{safe}.png")
        plot_heatmap(grid_df, label, f"grid_fixed_{safe}.png")
        grid_df.to_csv(f"grid_fixed_{safe}.csv", index=False)

        # ── BACK TEST 2: Nearest 6 months ──────────────────────────────────────
        print(f"\n── BACK TEST 2 (Nearest 6 months: {BT2_START} → {BT2_END}) ──")
        if len(df_bt2) > 30:
            ec_bt2, tr_bt2 = run_backtest(df_bt2, beta_bt2,
                                          capital=10_000, **bp_grid)
            bt2_m = compute_metrics(ec_bt2, tr_bt2, 10_000)
            print_metrics_table(bt2_m, label)
        else:
            bt2_m = {}
            print("  Insufficient data for BT2.")

        # ── SCENARIO TESTS ─────────────────────────────────────────────────────
        scenario_results = {}
        print(f"\n── SCENARIO TESTS ──")
        for scenario_name, (sc_start, sc_end) in SCENARIOS.items():
            df_sc = df_full[(df_full.index >= sc_start) &
                            (df_full.index <= sc_end)]
            if len(df_sc) < 50:
                print(f"  {scenario_name}: insufficient data")
                continue

            b_sc = beta_full[df_sc.index]
            ec_sc, tr_sc = run_backtest(df_sc, b_sc,
                                        capital=10_000, **bp_grid)
            sc_m = compute_metrics(ec_sc, tr_sc, 10_000)
            scenario_results[scenario_name] = sc_m
            print(f"\n  ── {scenario_name} ({sc_start} → {sc_end}) ──")
            print(f"  Sharpe={sc_m.get('sharpe_ratio','—')}  "
                  f"CAGR={sc_m.get('annualized_return_compounding','—')}%  "
                  f"MaxDD={sc_m.get('max_drawdown_pct','—')}%  "
                  f"WinRate={sc_m.get('win_rate_pct','—')}%  "
                  f"Trades={sc_m.get('num_trades','—')}")

        # ── Store all results ──────────────────────────────────────────────────
        all_output[label] = {
            "best_params":       bp_grid,
            "bt1_train":         train_m,
            "bt1_test":          test_m,
            "bt2_recent":        bt2_m,
            "scenarios":         scenario_results,
        }

    # ── Save ──────────────────────────────────────────────────────────────────
    with open("backtest_fixed_results.json", "w") as f:
        json.dump(all_output, f, indent=2, default=str)
    print(f"\n{'='*65}")
    print("All results saved → backtest_fixed_results.json")
    print("Charts saved as equity_fixed_*.png and grid_fixed_*.png")
    print("Grid CSVs saved as grid_fixed_*.csv")


if __name__ == "__main__":
    run_pipeline()