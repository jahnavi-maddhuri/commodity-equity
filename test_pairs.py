"""
Commodity-Equity Pair Testing
==============================
Tests candidate pairs for cointegration and Granger causality.
Run this locally: python pair_testing.py

Requirements:
    pip install yfinance statsmodels pandas numpy matplotlib seaborn
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from statsmodels.tsa.stattools import coint, grangercausalitytests
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller
from datetime import datetime


# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────

# Define candidate pairs: (commodity_etf, equity_ticker, label)
PAIRS = [
    ("HG=F",  "TEL",  "Copper → TE Connectivity"),
    ("HG=F", "ROK",  "Copper → Rockwell Automation"),

    # # FAILED: 
    # ("NG=F",  "CF",   "Natural Gas → CF Industries"),
    # ("NG=F",  "NTR",  "Natural Gas → Nutrien"),
    # ("CL=F",  "AAL",  "Crude Oil → American Airlines"),
    # ("CL=F",  "UAL",  "Crude Oil → United Airlines"),
    # ("HG=F",  "APTV", "Copper → Aptiv"),
    # ("CL=F",  "DAL",  "Crude Oil → Delta Airlines"),
    # ("HG=F", "ETN",  "Copper → Eaton Corp"),
    # ("HG=F", "EMR",  "Copper → Emerson Electric"), 
    # ("HG=F", "AME",  "Copper → Ametek"),           
    # ("HG=F", "PH",   "Copper → Parker Hannifin"),   
    # ("HG=F", "IR",   "Copper → Ingersoll Rand"),   
    # ("HG=F", "GE",   "Copper → GE Vernova"),       
    # ("LBS=F", "LEN",  "Lumber → Lennar"),
    # ("LBS=F", "DHI",  "Lumber → D.R. Horton"),
    # ("LBS=F", "PHM",  "Lumber → PulteGroup"),
    # ("HG=F", "FCX", "Copper → Freeport McMoRan (Producer)"),
    # ("HG=F", "HUBB", "Copper → Hubbell"),
]

START_DATE    = "2017-01-01"
END_DATE      = "2020-03-01"
GRANGER_LAGS  = 10          # Max lags to test for Granger causality
COINT_PVALUE  = 0.05        # Significance threshold for cointegration
GRANGER_PVALUE = 0.05       # Significance threshold for Granger causality


# ─────────────────────────────────────────────
# 2. DATA COLLECTION
# ─────────────────────────────────────────────

def download_pair(commodity_ticker, equity_ticker, start, end):
    """
    Downloads daily closing prices for one commodity-equity pair.
    Returns a cleaned DataFrame with columns [commodity, equity].
    """
    print(f"  Downloading {commodity_ticker} and {equity_ticker}...")

    commodity_data = yf.download(
        commodity_ticker, start=start, end=end,
        progress=False, auto_adjust=True
    )["Close"]

    equity_data = yf.download(
        equity_ticker, start=start, end=end,
        progress=False, auto_adjust=True
    )["Close"]

    # Flatten MultiIndex columns if present
    if isinstance(commodity_data, pd.DataFrame):
        commodity_data = commodity_data.iloc[:, 0]
    if isinstance(equity_data, pd.DataFrame):
        equity_data = equity_data.iloc[:, 0]

    df = pd.DataFrame({
        "commodity": commodity_data,
        "equity":    equity_data
    })
    df = df.ffill().dropna()

    return df


# ─────────────────────────────────────────────
# 3. STATISTICAL TESTS
# ─────────────────────────────────────────────

def test_stationarity(series, name):
    """
    Augmented Dickey-Fuller test.
    A price series should be non-stationary (p > 0.05).
    The spread should be stationary (p < 0.05).
    """
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "series":    name,
        "adf_stat":  round(result[0], 4),
        "p_value":   round(result[1], 4),
        "stationary": result[1] < 0.05
    }


def test_cointegration(df):
    """
    Engle-Granger cointegration test.
    H0: No cointegration (spread is non-stationary / no mean reversion).
    Reject H0 (p < 0.05) → pair is cointegrated → tradeable.
    """
    score, pvalue, _ = coint(df["equity"], df["commodity"])
    return {
        "test":       "Engle-Granger Cointegration",
        "score":      round(score, 4),
        "p_value":    round(pvalue, 4),
        "passed":     pvalue < COINT_PVALUE,
        "conclusion": "PASS ✓ — Mean-reverting spread confirmed"
            if pvalue < COINT_PVALUE
            else "FAIL ✗ — No reliable mean reversion"
    }


def test_granger_causality(df, max_lags=GRANGER_LAGS):
    """
    Granger causality test: does commodity LEAD equity?
    Tests H0: commodity does NOT Granger-cause equity.
    Reject H0 (p < 0.05) at some lag → commodity leads equity → signal is valid.

    Returns best lag (strongest causality) and whether the test passed.
    """
    returns = pd.DataFrame({
        "equity":    df["equity"].pct_change(),
        "commodity": df["commodity"].pct_change()
    }).dropna()

    # Commodity → Equity direction (this is what we want)
    results = grangercausalitytests(
        returns[["equity", "commodity"]],
        maxlag=max_lags,
        verbose=False
    )

    # Find the lag with the lowest p-value
    best_lag    = None
    best_pvalue = 1.0
    lag_summary = {}

    for lag, result in results.items():
        pval = result[0]["ssr_ftest"][1]   # F-test p-value
        lag_summary[lag] = round(pval, 4)
        if pval < best_pvalue:
            best_pvalue = pval
            best_lag    = lag

    passed = best_pvalue < GRANGER_PVALUE

    return {
        "test":       "Granger Causality (Commodity → Equity)",
        "best_lag":   best_lag,
        "best_pvalue": round(best_pvalue, 4),
        "all_lags":   lag_summary,
        "passed":     passed,
        "conclusion": f"PASS ✓ — Commodity leads equity at lag {best_lag} days"
            if passed
            else "FAIL ✗ — No clear directional lead from commodity"
    }


def estimate_beta(df, window=252):
    """
    OLS regression: equity returns ~ commodity returns.
    Estimates the commodity-beta (sensitivity).
    Returns full-sample beta and a rolling beta series.
    """
    # returns = pd.DataFrame({
    #     "equity":    df["equity"].pct_change(),
    #     "commodity": df["commodity"].pct_change()
    # }).dropna()
    returns = pd.DataFrame({
        "equity":    df["equity"].pct_change(),
        "commodity": df["commodity"].pct_change()
    }).resample("W").sum().dropna()   # Weekly returns

    # Full-sample beta
    X     = add_constant(returns["commodity"])
    model = OLS(returns["equity"], X).fit()
    beta  = model.params["commodity"]
    alpha = model.params["const"]
    r2    = model.rsquared

    # Rolling beta
    rolling_beta = (
        returns["equity"]
        .rolling(window)
        .cov(returns["commodity"])
        / returns["commodity"]
        .rolling(window)
        .var()
    )

    return {
        "beta":         round(beta, 4),
        "alpha":        round(alpha, 6),
        "r_squared":    round(r2, 4),
        "rolling_beta": rolling_beta
    }


def compute_spread_and_zscore(df, beta, signal_window=10, zscore_window=60):
    """
    Computes the mispricing spread and its rolling Z-score.

    Spread = (beta × commodity_return) - equity_return
    Positive spread → equity underreacted to commodity rise → short equity
    Negative spread → equity underreacted to commodity fall → long equity
    """
    returns = pd.DataFrame({
        "equity":    df["equity"].pct_change(signal_window),
        "commodity": df["commodity"].pct_change(signal_window)
    }).dropna()

    returns["spread"] = (beta * returns["commodity"]) - returns["equity"]

    roll_mean = returns["spread"].rolling(zscore_window).mean()
    roll_std  = returns["spread"].rolling(zscore_window).std()

    returns["zscore"] = (returns["spread"] - roll_mean) / roll_std

    return returns


# ─────────────────────────────────────────────
# 4. VISUALIZATION
# ─────────────────────────────────────────────

def plot_pair_analysis(df, spread_df, beta_info, coint_result,
                       granger_result, label, save_path=None):
    """
    Produces a 4-panel diagnostic chart for one pair:
    1. Normalized price series
    2. Rolling beta over time
    3. Spread over time
    4. Z-score with entry/exit thresholds
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Pair Analysis: {label}", fontsize=14, fontweight="bold", y=0.98)
    gs  = gridspec.GridSpec(4, 1, hspace=0.45)

    color_commodity = "#E07B39"
    color_equity    = "#3A7EBF"
    color_spread    = "#5B5EA6"
    color_zscore    = "#2E8B57"

    # ── Panel 1: Normalized prices ──
    ax1 = fig.add_subplot(gs[0])
    norm_c = df["commodity"] / df["commodity"].iloc[0] * 100
    norm_e = df["equity"]    / df["equity"].iloc[0]    * 100
    ax1.plot(norm_c.index, norm_c, color=color_commodity,
             label="Commodity (ETF)", linewidth=1.2)
    ax1.plot(norm_e.index, norm_e, color=color_equity,
             label="Equity", linewidth=1.2)
    ax1.set_title("Normalized Prices (Base = 100)", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.set_ylabel("Price (indexed)")
    ax1.grid(alpha=0.3)

    # ── Panel 2: Rolling beta ──
    ax2 = fig.add_subplot(gs[1])
    rb  = beta_info["rolling_beta"].dropna()
    ax2.plot(rb.index, rb, color=color_commodity, linewidth=1.0)
    ax2.axhline(beta_info["beta"], color="black", linestyle="--",
                linewidth=0.8, label=f"Full-sample β = {beta_info['beta']:.3f}")
    ax2.axhline(0, color="gray", linestyle=":", linewidth=0.6)
    ax2.set_title(f"Rolling 252-day Beta  |  R² = {beta_info['r_squared']:.3f}",
                  fontsize=10)
    ax2.legend(fontsize=8)
    ax2.set_ylabel("Beta")
    ax2.grid(alpha=0.3)

    # ── Panel 3: Spread ──
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(spread_df.index, spread_df["spread"],
             color=color_spread, linewidth=1.0)
    ax3.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax3.fill_between(spread_df.index, spread_df["spread"], 0,
                     where=spread_df["spread"] > 0,
                     alpha=0.2, color="#E07B39", label="Short signal zone")
    ax3.fill_between(spread_df.index, spread_df["spread"], 0,
                     where=spread_df["spread"] < 0,
                     alpha=0.2, color="#3A7EBF", label="Long signal zone")
    ax3.set_title("Mispricing Spread  (β × ΔCommodity − ΔEquity)", fontsize=10)
    ax3.legend(fontsize=8)
    ax3.set_ylabel("Spread")
    ax3.grid(alpha=0.3)

    # ── Panel 4: Z-score with thresholds ──
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(spread_df.index, spread_df["zscore"],
             color=color_zscore, linewidth=1.0, label="Z-score")
    ax4.axhline( 1.5, color="#E07B39", linestyle="--",
                linewidth=1.0, label="+1.5σ Short entry")
    ax4.axhline(-1.5, color="#3A7EBF", linestyle="--",
                linewidth=1.0, label="−1.5σ Long entry")
    ax4.axhline( 0.25, color="green",  linestyle=":",
                linewidth=0.8, label="±0.25σ Exit")
    ax4.axhline(-0.25, color="green",  linestyle=":", linewidth=0.8)
    ax4.axhline(0,     color="black",  linestyle="-",  linewidth=0.5)
    ax4.fill_between(spread_df.index, 1.5,
                     spread_df["zscore"].clip(lower=1.5),
                     alpha=0.25, color="#E07B39")
    ax4.fill_between(spread_df.index, -1.5,
                     spread_df["zscore"].clip(upper=-1.5),
                     alpha=0.25, color="#3A7EBF")
    ax4.set_title("Z-Score of Spread  |  Entry ±1.5σ  |  Exit ±0.25σ",
                  fontsize=10)
    ax4.legend(fontsize=7, ncol=3)
    ax4.set_ylabel("Z-Score")
    ax4.grid(alpha=0.3)

    # ── Test result annotation ──
    coint_str   = f"Cointegration p={coint_result['p_value']}  "   \
                  f"{'✓' if coint_result['passed'] else '✗'}"
    granger_str = f"Granger p={granger_result['best_pvalue']} "    \
                  f"lag={granger_result['best_lag']}d  "           \
                  f"{'✓' if granger_result['passed'] else '✗'}"
    fig.text(0.01, 0.01,
             f"{coint_str}     {granger_str}",
             fontsize=8, color="gray",
             verticalalignment="bottom")

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Chart saved → {save_path}")
    else:
        plt.show()

    plt.close()


# ─────────────────────────────────────────────
# 5. MAIN RUNNER
# ─────────────────────────────────────────────

def run_all_pairs():
    """
    Runs the full data collection + statistical testing pipeline
    for every pair defined in PAIRS. Prints a summary scorecard.
    """
    results_summary = []

    print("=" * 60)
    print("COMMODITY-EQUITY PAIR TESTING PIPELINE")
    print(f"Period: {START_DATE} → {END_DATE}")
    print("=" * 60)

    for commodity_ticker, equity_ticker, label in PAIRS:
        print(f"\n{'─' * 60}")
        print(f"PAIR: {label}")
        print(f"{'─' * 60}")

        # ── Download data ──
        try:
            df = download_pair(commodity_ticker, equity_ticker,
                               START_DATE, END_DATE)
        except Exception as e:
            print(f"  DATA ERROR: {e}")
            continue

        if len(df) < 500:
            print(f"  SKIPPED — insufficient data ({len(df)} rows)")
            continue

        print(f"  Data: {len(df)} trading days  "
              f"({df.index[0].date()} → {df.index[-1].date()})")

        # ── Stationarity checks (informational) ──
        adf_c = test_stationarity(df["commodity"], commodity_ticker)
        adf_e = test_stationarity(df["equity"],    equity_ticker)
        print(f"  ADF — {commodity_ticker}: p={adf_c['p_value']} "
              f"({'stationary' if adf_c['stationary'] else 'non-stationary ✓'})")
        print(f"  ADF — {equity_ticker}:  p={adf_e['p_value']} "
              f"({'stationary' if adf_e['stationary'] else 'non-stationary ✓'})")

        # ── Cointegration ──
        coint_result = test_cointegration(df)
        print(f"  {coint_result['test']}: "
              f"p={coint_result['p_value']} → {coint_result['conclusion']}")

        # ── Granger causality ──
        granger_result = test_granger_causality(df)
        print(f"  {granger_result['test']}: "
              f"best p={granger_result['best_pvalue']} "
              f"at lag {granger_result['best_lag']}d "
              f"→ {granger_result['conclusion']}")

        # ── Beta estimation ──
        beta_info = estimate_beta(df)
        print(f"  Beta (β): {beta_info['beta']}  |  "
              f"R²: {beta_info['r_squared']}  |  "
              f"Alpha: {beta_info['alpha']}")

        # ── Spread and Z-score ──
        spread_df = compute_spread_and_zscore(df, beta_info["beta"])

        # ── Plot ──
        safe_label = label.replace(" ", "_").replace("→", "to").replace("/", "-")
        chart_path = f"chart_{safe_label}.png"
        plot_pair_analysis(
            df, spread_df, beta_info,
            coint_result, granger_result,
            label, save_path=chart_path
        )

        # ── Score the pair ──
        both_passed = coint_result["passed"] and granger_result["passed"]
        score       = sum([
            coint_result["passed"],
            granger_result["passed"],
            beta_info["r_squared"] > 0.1,
            beta_info["beta"] < 0,   # Negative beta = rising commodity hurts equity
        ])

        results_summary.append({
            "Pair":              label,
            "Data Points":       len(df),
            "Coint p-value":     coint_result["p_value"],
            "Coint Pass":        "✓" if coint_result["passed"] else "✗",
            "Granger p-value":   granger_result["best_pvalue"],
            "Granger Lag (days)":granger_result["best_lag"],
            "Granger Pass":      "✓" if granger_result["passed"] else "✗",
            "Beta (β)":          beta_info["beta"],
            "R²":                beta_info["r_squared"],
            "Both Tests Pass":   "✓✓ TRADE" if both_passed else "✗  SKIP",
            "Score":             f"{score}/4"
        })

    # ── Print scorecard ──
    print(f"\n{'=' * 60}")
    print("PAIR SCORECARD SUMMARY")
    print(f"{'=' * 60}")

    summary_df = pd.DataFrame(results_summary)

    if summary_df.empty:
        print("No pairs successfully tested.")
        return summary_df

    # Sort by both tests passing first, then Granger p-value
    summary_df["_sort"] = summary_df["Both Tests Pass"].apply(
        lambda x: 0 if "TRADE" in x else 1
    )
    summary_df = summary_df.sort_values(["_sort", "Granger p-value"])
    summary_df = summary_df.drop(columns=["_sort"])

    # Print readable table
    print_cols = ["Pair", "Coint Pass", "Coint p-value",
                  "Granger Pass", "Granger Lag (days)",
                  "Beta (β)", "R²", "Both Tests Pass"]
    print(summary_df[print_cols].to_string(index=False))

    # Save to CSV
    summary_df.to_csv("pair_test_results.csv", index=False)
    print("\nFull results saved → pair_test_results.csv")
    print("Charts saved as chart_*.png")

    return summary_df


# ─────────────────────────────────────────────
# 6. ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    results = run_all_pairs()