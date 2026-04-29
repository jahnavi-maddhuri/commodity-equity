"""
Tenor Comparison — Stages 1 & 2
============================================================
Stage 1: Data collection and diagnostics.
Stage 2: Cointegration, Granger causality, OLS β/R² across
         (equity × deferred contract × signal type × window),
         plus a separate long-horizon lagged-regression block.

Ticker note: yfinance does not carry individual deferred copper
contracts under the standard "=F" suffix. They are available under
the COMEX exchange suffix ".CMX". Front-month remains HG=F (continuous).

Caveat on tenor labeling: tenor drifts within a single contract series
(e.g., HGZ26 was ~24 months out a year ago, ~8 months out today). Mean
days-to-expiry labels smooth over this drift; the regression
observations themselves span varying tenors within each contract's
window.

Caveat on regime concentration: the deferred contracts share roughly
2022–2025 as a common window. Cross-regime conclusions cannot be drawn
from stage 2 results alone — that's a writeup point.

Stage 1 outputs:
    stage1_raw_data.pkl       — pickled dict of {ticker: pd.Series}
    stage1_contracts.csv      — per-contract availability
    stage1_pairings.csv       — deferred × equity overlap
    stage1_spreads.csv        — calendar spread ADF

Stage 2 outputs:
    tenor_comparison_results.csv  — main test grid
    tenor_lagged_regression.csv   — long-horizon lead-lag profile
"""

import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import re
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint, grangercausalitytests
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.multitest import multipletests

from test_pairs import test_stationarity


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
FRONT_MONTH = "HG=F"
DEFERRED    = ["HGN26.CMX", "HGU26.CMX", "HGZ25.CMX", "HGZ26.CMX", "HGZ27.CMX"]
CONTRACTS   = [FRONT_MONTH] + DEFERRED
EQUITIES    = ["ROK", "TEL"]

MIN_OVERLAP_ROWS = 60
GAP_TD_THRESHOLD = 5
FFILL_LIMIT      = 2

# Stage 2 test parameters
WINDOWS_NEW         = [30, 60, 90]
GRANGER_MAXLAG_NEW  = 130
GRANGER_TARGET_LAGS = [5, 20, 60, 90, 130]

# Original baseline parameters (from test_pairs.py)
BASELINE_START      = "2017-01-01"
BASELINE_END        = "2020-03-01"
BASELINE_WINDOW     = 15
BASELINE_MAXLAG     = 10
BASELINE_TARGET_LAGS = [5, 20, 60, 90, 130]   # most will be NaN at maxlag=10

# Lagged regression
LAGGED_K_VALUES = [5, 20, 60, 90, 130, 180, 252]

# Significance threshold
ALPHA = 0.05

# Paths
RAW_DATA_PATH       = "stage1_raw_data.pkl"
CONTRACTS_CSV_PATH  = "stage1_contracts.csv"
PAIRINGS_CSV_PATH   = "stage1_pairings.csv"
SPREADS_CSV_PATH    = "stage1_spreads.csv"
MAIN_RESULTS_PATH   = "tenor_comparison_results.csv"
LAGGED_RESULTS_PATH = "tenor_lagged_regression.csv"

# Futures month codes
MONTH_CODES = {"F":1, "G":2, "H":3, "J":4, "K":5, "M":6,
               "N":7, "Q":8, "U":9, "V":10, "X":11, "Z":12}


# ─────────────────────────────────────────────
# Data pull
# ─────────────────────────────────────────────
def download_max(ticker):
    print(f"  Downloading {ticker} (period=max)...")
    try:
        df = yf.download(ticker, period="max", progress=False, auto_adjust=True)
    except Exception as e:
        print(f"    ERROR: {e}")
        return pd.Series(dtype=float, name=ticker)
    if df is None or df.empty:
        print(f"    yfinance returned no data for {ticker}")
        return pd.Series(dtype=float, name=ticker)
    s = df["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = s.dropna()
    s.name = ticker
    return s


def pull_or_load_data():
    """Load cached raw data if pickled; otherwise pull from yfinance."""
    if os.path.exists(RAW_DATA_PATH):
        with open(RAW_DATA_PATH, "rb") as f:
            raw = pickle.load(f)
        print(f"Loaded raw data from {RAW_DATA_PATH} "
              f"(delete to force re-pull)")
        return raw
    print("Pulling raw data from yfinance...")
    raw = {}
    for t in CONTRACTS + EQUITIES:
        raw[t] = download_max(t)
    with open(RAW_DATA_PATH, "wb") as f:
        pickle.dump(raw, f)
    print(f"Saved raw data → {RAW_DATA_PATH}")
    return raw


# ─────────────────────────────────────────────
# Tenor labeling
# ─────────────────────────────────────────────
def contract_expiry_date(ticker):
    """Parse ticker (e.g., 'HGZ26.CMX') → expiry date proxy
    (last business day of contract month).
    Returns None for non-deferred tickers."""
    m = re.match(r"HG([FGHJKMNQUVXZ])(\d{2})\.CMX", ticker)
    if not m:
        return None
    month = MONTH_CODES[m.group(1)]
    yr2 = int(m.group(2))
    year = 2000 + yr2 if yr2 < 80 else 1900 + yr2
    last_cal = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
    last_bday = last_cal - pd.offsets.BDay(0) if last_cal.weekday() < 5 \
                else last_cal - pd.offsets.BDay(1)
    return last_bday


def mean_tenor_months(ticker, dates):
    """Mean days-to-expiry over `dates`, expressed in months (≈30.44 days)."""
    expiry = contract_expiry_date(ticker)
    if expiry is None or len(dates) == 0:
        return None
    dte_days = (expiry - pd.DatetimeIndex(dates)).days
    return round(float(np.mean(dte_days) / 30.44), 2)


# ─────────────────────────────────────────────
# Stage 1 — diagnostics
# ─────────────────────────────────────────────
def find_gaps(series, threshold_td=GAP_TD_THRESHOLD):
    if len(series) < 2:
        return []
    idx = series.index
    cal_diffs = idx.to_series().diff().dt.days
    gaps = []
    for i in range(1, len(idx)):
        cd = cal_diffs.iloc[i]
        if pd.isna(cd) or cd <= 3:
            continue
        bdays_between = len(pd.bdate_range(idx[i-1], idx[i])) - 2
        if bdays_between > threshold_td:
            gaps.append((idx[i-1].date(), idx[i].date(),
                         int(bdays_between), int(cd)))
    return gaps


def diagnose_contract(ticker, series):
    n = len(series)
    if n == 0:
        return {"ticker": ticker, "n_rows": 0,
                "first_date": None, "last_date": None,
                "n_gaps_gt_5td": 0, "max_gap_td": None,
                "max_gap_cal_days": None, "_gaps": []}
    gaps = find_gaps(series)
    return {"ticker": ticker, "n_rows": n,
            "first_date": series.index[0].date(),
            "last_date": series.index[-1].date(),
            "n_gaps_gt_5td": len(gaps),
            "max_gap_td": max((g[2] for g in gaps), default=None),
            "max_gap_cal_days": max((g[3] for g in gaps), default=None),
            "_gaps": gaps}


def diagnose_pairing(deferred_series, equity_series):
    if len(deferred_series) == 0 or len(equity_series) == 0:
        return {"n_overlap_raw": 0, "n_overlap_ffilled": 0,
                "first_date": None, "last_date": None,
                "passes_60_gate": False}
    df = pd.concat([deferred_series, equity_series], axis=1, join="inner")
    df.columns = ["deferred", "equity"]
    n_raw = df.dropna().shape[0]
    df_filled = df.ffill(limit=FFILL_LIMIT).dropna()
    n_filled = len(df_filled)
    return {"n_overlap_raw": n_raw, "n_overlap_ffilled": n_filled,
            "first_date": df_filled.index[0].date() if n_filled > 0 else None,
            "last_date":  df_filled.index[-1].date() if n_filled > 0 else None,
            "passes_60_gate": n_filled >= MIN_OVERLAP_ROWS}


def diagnose_spread(deferred_series, front_series):
    if len(deferred_series) == 0 or len(front_series) == 0:
        return {"n_spread_rows": 0, "first_date": None, "last_date": None,
                "spread_mean": None, "spread_std": None,
                "adf_p": None, "stationary": None}
    df = pd.concat([deferred_series, front_series], axis=1, join="inner")
    df.columns = ["deferred", "front"]
    df = df.ffill(limit=FFILL_LIMIT).dropna()
    n = len(df)
    if n < 30:
        return {"n_spread_rows": n,
                "first_date": df.index[0].date() if n > 0 else None,
                "last_date":  df.index[-1].date() if n > 0 else None,
                "spread_mean": None, "spread_std": None,
                "adf_p": None, "stationary": None}
    spread = df["deferred"] - df["front"]
    adf = test_stationarity(spread, "spread")
    return {"n_spread_rows": n,
            "first_date": df.index[0].date(),
            "last_date":  df.index[-1].date(),
            "spread_mean": round(float(spread.mean()), 4),
            "spread_std":  round(float(spread.std()), 4),
            "adf_p":       adf["p_value"],
            "stationary":  bool(adf["stationary"])}


def stage1_diagnostics(raw):
    print("\n" + "=" * 72)
    print("STAGE 1 — DATA COLLECTION AND DIAGNOSTICS")
    print("=" * 72)

    print("\n[Contracts]")
    contract_diag = [diagnose_contract(t, raw[t]) for t in CONTRACTS]
    for d in contract_diag:
        if d["n_rows"] == 0:
            print(f"  {d['ticker']:12s}  EMPTY"); continue
        gap_str = (f"  gaps>{GAP_TD_THRESHOLD}td: {d['n_gaps_gt_5td']}"
                   + (f" (max {d['max_gap_td']}td / {d['max_gap_cal_days']}cal)"
                      if d["max_gap_td"] else ""))
        print(f"  {d['ticker']:12s}  rows={d['n_rows']:5d}  "
              f"{d['first_date']} → {d['last_date']}{gap_str}")

    print("\n[Equities]")
    for t in EQUITIES:
        d = diagnose_contract(t, raw[t])
        print(f"  {t:12s}  rows={d['n_rows']:5d}  "
              f"{d['first_date']} → {d['last_date']}")

    print("\n[Deferred × equity overlap]")
    pairing_diag = []
    for d_tkr in DEFERRED:
        for eq in EQUITIES:
            p = diagnose_pairing(raw[d_tkr], raw[eq])
            pairing_diag.append({"deferred": d_tkr, "equity": eq, **p})
            gate = "PASS" if p["passes_60_gate"] else "SKIP"
            print(f"  {d_tkr:12s} × {eq:5s}  n={p['n_overlap_ffilled']:5d}  "
                  f"{p['first_date']} → {p['last_date']}  [{gate}]")

    print("\n[Calendar spread ADF]")
    spread_diag = []
    for d_tkr in DEFERRED:
        s = diagnose_spread(raw[d_tkr], raw[FRONT_MONTH])
        spread_diag.append({"deferred": d_tkr, **s})
        if s["n_spread_rows"] < 30:
            print(f"  {d_tkr:12s}  insufficient"); continue
        flag = "STATIONARY" if s["stationary"] else "non-stationary"
        print(f"  {d_tkr:12s}  n={s['n_spread_rows']:5d}  "
              f"ADF p={s['adf_p']:.4f}  [{flag} at α={ALPHA}]")

    pd.DataFrame([{k: v for k, v in d.items() if not k.startswith("_")}
                  for d in contract_diag]).to_csv(CONTRACTS_CSV_PATH, index=False)
    pd.DataFrame(pairing_diag).to_csv(PAIRINGS_CSV_PATH, index=False)
    pd.DataFrame(spread_diag).to_csv(SPREADS_CSV_PATH, index=False)


# ─────────────────────────────────────────────
# Stage 2 — statistical tests
# ─────────────────────────────────────────────
def merge_pair(commodity_series, equity_series, date_range=None):
    """Inner-join, ffill ≤2, dropna. Optionally restrict to a date range."""
    df = pd.concat([commodity_series, equity_series], axis=1, join="inner")
    df.columns = ["commodity", "equity"]
    if date_range is not None:
        df = df.loc[date_range[0]:date_range[1]]
    df = df.ffill(limit=FFILL_LIMIT).dropna()
    return df


def cointegration_p(df):
    eq = df["equity"].replace([np.inf, -np.inf], np.nan).dropna()
    co = df["commodity"].replace([np.inf, -np.inf], np.nan).dropna()
    aligned = pd.concat([eq, co], axis=1, join="inner").dropna()
    if len(aligned) < 30:
        return None
    try:
        _score, pvalue, _crit = coint(aligned["equity"], aligned["commodity"])
        return float(pvalue)
    except Exception as e:
        print(f"    coint failed: {e}")
        return None


def estimate_beta_windowed(df, window):
    """Windowed pct_change → OLS β, R²."""
    returns = pd.DataFrame({
        "equity":    df["equity"].pct_change(window),
        "commodity": df["commodity"].pct_change(window)
    }).replace([np.inf, -np.inf], np.nan).dropna()
    if (len(returns) < 10
            or returns["commodity"].std() == 0
            or returns["equity"].std() == 0):
        return {"beta": None, "r_squared": None, "n_obs_beta": len(returns)}
    try:
        X = add_constant(returns["commodity"])
        model = OLS(returns["equity"], X).fit()
        return {"beta": float(model.params["commodity"]),
                "r_squared": float(model.rsquared),
                "n_obs_beta": len(returns)}
    except Exception as e:
        print(f"    OLS failed: {e}")
        return {"beta": None, "r_squared": None, "n_obs_beta": len(returns)}


def granger_with_target_lags(df, maxlag, target_lags):
    """Daily pct_change → Granger F-test p-values at requested lags."""
    returns = pd.DataFrame({
        "equity":    df["equity"].pct_change(),
        "commodity": df["commodity"].pct_change()
    }).replace([np.inf, -np.inf], np.nan).dropna()

    out = {f"granger_p_lag{k}": None for k in target_lags}
    out.update({"granger_best_lag": None, "granger_best_p": None,
                "granger_passed": False, "granger_maxlag_used": None,
                "n_obs_granger": len(returns)})

    if len(returns) < 3 * maxlag + 5:
        return out

    try:
        results = grangercausalitytests(
            returns[["equity", "commodity"]],
            maxlag=maxlag, verbose=False
        )
    except Exception as e:
        print(f"    granger failed at maxlag={maxlag}: {e}")
        return out

    lag_pvals = {lag: float(res[0]["ssr_ftest"][1])
                 for lag, res in results.items()}
    for k in target_lags:
        out[f"granger_p_lag{k}"] = lag_pvals.get(k)
    best_lag = min(lag_pvals, key=lag_pvals.get)
    out["granger_best_lag"] = int(best_lag)
    out["granger_best_p"]   = lag_pvals[best_lag]
    out["granger_passed"]   = lag_pvals[best_lag] < ALPHA
    out["granger_maxlag_used"] = maxlag
    return out


def spread_adf_p(spread_series):
    if len(spread_series.dropna()) < 30:
        return None
    return float(test_stationarity(spread_series.dropna(), "spread")["p_value"])


def make_test_row(*, equity, commodity_label, commodity_series, equity_series,
                  window, maxlag, target_lags, tenor_months, date_range=None,
                  spread_series_for_adf=None):
    """Run all tests for one row of the main results table."""
    df = merge_pair(commodity_series, equity_series, date_range=date_range)
    n_obs = len(df)

    coint_p = cointegration_p(df) if n_obs >= 30 else None
    beta_info = estimate_beta_windowed(df, window) if n_obs >= window + 10 \
                else {"beta": None, "r_squared": None, "n_obs_beta": 0}
    g = granger_with_target_lags(df, maxlag=maxlag, target_lags=target_lags)

    adf_p = (spread_adf_p(spread_series_for_adf)
             if spread_series_for_adf is not None else None)

    return {
        "equity": equity,
        "commodity_series": commodity_label,
        "tenor_months_actual": tenor_months,
        "window_days": window,
        "data_window_start": df.index[0].date() if n_obs > 0 else None,
        "data_window_end":   df.index[-1].date() if n_obs > 0 else None,
        "n_observations": n_obs,
        "coint_p": coint_p,
        "coint_passed": (coint_p is not None and coint_p < ALPHA),
        "spread_adf_p": adf_p,
        "granger_maxlag_used": g["granger_maxlag_used"],
        "granger_p_lag5":   g["granger_p_lag5"],
        "granger_p_lag20":  g["granger_p_lag20"],
        "granger_p_lag60":  g["granger_p_lag60"],
        "granger_p_lag90":  g["granger_p_lag90"],
        "granger_p_lag130": g["granger_p_lag130"],
        "granger_best_lag":  g["granger_best_lag"],
        "granger_best_p":    g["granger_best_p"],
        "granger_passed":    g["granger_passed"],
        "beta": beta_info["beta"],
        "r_squared": beta_info["r_squared"],
    }


def build_main_results(raw):
    """Build the main test grid:
        - 2 original baselines (HG=F vs {ROK,TEL}, 2017-2020, w=15, maxlag=10)
        - For each deferred × equity × window:
            - level row
            - spread row
            - matched front-month baseline row (HG=F on deferred's date range)
    """
    rows = []
    front = raw[FRONT_MONTH]

    # ── Original baselines ──
    for eq in EQUITIES:
        eq_series = raw[eq]
        rows.append(make_test_row(
            equity=eq,
            commodity_label="HG=F front-month (original baseline)",
            commodity_series=front,
            equity_series=eq_series,
            window=BASELINE_WINDOW,
            maxlag=BASELINE_MAXLAG,
            target_lags=BASELINE_TARGET_LAGS,
            tenor_months=1.0,
            date_range=(BASELINE_START, BASELINE_END),
        ))
        print(f"  baseline (orig)        HG=F × {eq:5s}  done")

    # ── Per-deferred rows ──
    for d_tkr in DEFERRED:
        deferred_series = raw[d_tkr]
        # spread series aligned on deferred×front overlap
        spread_df = pd.concat([deferred_series, front],
                              axis=1, join="inner").ffill(limit=FFILL_LIMIT).dropna()
        spread_df.columns = ["deferred", "front"]
        spread_series = (spread_df["deferred"] - spread_df["front"]).rename("spread")

        # Mean tenor from deferred's date range
        tenor_def = mean_tenor_months(d_tkr, deferred_series.index)

        for eq in EQUITIES:
            eq_series = raw[eq]
            # Date range = the deferred's full overlap with itself (its own index)
            d_range = (deferred_series.index[0], deferred_series.index[-1])

            for window in WINDOWS_NEW:
                # Level row
                rows.append(make_test_row(
                    equity=eq,
                    commodity_label=f"{d_tkr} level",
                    commodity_series=deferred_series,
                    equity_series=eq_series,
                    window=window,
                    maxlag=GRANGER_MAXLAG_NEW,
                    target_lags=GRANGER_TARGET_LAGS,
                    tenor_months=tenor_def,
                ))
                # Spread row
                rows.append(make_test_row(
                    equity=eq,
                    commodity_label=f"{d_tkr} spread",
                    commodity_series=spread_series,
                    equity_series=eq_series,
                    window=window,
                    maxlag=GRANGER_MAXLAG_NEW,
                    target_lags=GRANGER_TARGET_LAGS,
                    tenor_months=tenor_def,
                    spread_series_for_adf=spread_series,
                ))
                # Matched front-month baseline (HG=F on deferred's date range)
                rows.append(make_test_row(
                    equity=eq,
                    commodity_label=f"HG=F front-month (matched {d_tkr} window)",
                    commodity_series=front,
                    equity_series=eq_series,
                    window=window,
                    maxlag=GRANGER_MAXLAG_NEW,
                    target_lags=GRANGER_TARGET_LAGS,
                    tenor_months=1.0,
                    date_range=d_range,
                ))
            print(f"  deferred {d_tkr:12s} × {eq:5s}  done "
                  f"(level/spread/matched-baseline × {len(WINDOWS_NEW)} windows)")

    df = pd.DataFrame(rows)

    # ── Benjamini-Hochberg FDR on coint_p across all rows ──
    pvals = df["coint_p"].values.astype(float)
    mask = ~np.isnan(pvals)
    adj = np.full_like(pvals, np.nan, dtype=float)
    if mask.any():
        _rej, adj_p, _, _ = multipletests(pvals[mask], alpha=ALPHA, method="fdr_bh")
        adj[mask] = adj_p
    df["coint_p_bh_adjusted"] = adj

    # ── Reorder columns ──
    col_order = [
        "equity", "commodity_series", "tenor_months_actual", "window_days",
        "data_window_start", "data_window_end", "n_observations",
        "coint_p", "coint_passed", "coint_p_bh_adjusted",
        "spread_adf_p", "granger_maxlag_used",
        "granger_p_lag5", "granger_p_lag20", "granger_p_lag60",
        "granger_p_lag90", "granger_p_lag130",
        "granger_best_lag", "granger_best_p", "granger_passed",
        "beta", "r_squared",
    ]
    df = df[col_order]
    return df


# ─────────────────────────────────────────────
# Stage 2 — long-horizon lagged regression
# ─────────────────────────────────────────────
def lagged_regression_row(equity, commodity_label, commodity_series,
                          equity_series, lag_k):
    """Regress equity_return(t) ~ commodity_return(t-k). Daily returns, single lag."""
    eq_ret = equity_series.pct_change()
    co_ret = commodity_series.pct_change()
    aligned = pd.DataFrame({
        "y": eq_ret,
        "x": co_ret.shift(lag_k)
    }).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(aligned)
    if n < 30:
        return {"equity": equity, "commodity_series": commodity_label,
                "lag_days": lag_k, "beta": None, "t_stat": None,
                "p_value": None, "n_observations": n}
    X = add_constant(aligned["x"])
    model = OLS(aligned["y"], X).fit()
    return {
        "equity": equity,
        "commodity_series": commodity_label,
        "lag_days": lag_k,
        "beta":   float(model.params["x"]),
        "t_stat": float(model.tvalues["x"]),
        "p_value": float(model.pvalues["x"]),
        "n_observations": n,
    }


def build_lagged_regression(raw):
    """For each (equity, commodity_series) pair, run lagged regression at
    all k in LAGGED_K_VALUES. Each pair uses its full natural overlap."""
    rows = []
    front = raw[FRONT_MONTH]

    series_specs = [("HG=F front-month", front)]
    for d_tkr in DEFERRED:
        deferred_series = raw[d_tkr]
        series_specs.append((f"{d_tkr} level", deferred_series))
        spread_df = pd.concat([deferred_series, front], axis=1, join="inner") \
                      .ffill(limit=FFILL_LIMIT).dropna()
        spread_df.columns = ["deferred", "front"]
        spread = (spread_df["deferred"] - spread_df["front"]).rename(f"{d_tkr}_spread")
        series_specs.append((f"{d_tkr} spread", spread))

    for eq in EQUITIES:
        eq_series = raw[eq]
        for label, comm_series in series_specs:
            for k in LAGGED_K_VALUES:
                rows.append(lagged_regression_row(eq, label, comm_series,
                                                  eq_series, k))
        print(f"  lagged regressions for {eq:5s}: "
              f"{len(series_specs)} commodity series × {len(LAGGED_K_VALUES)} lags")

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    raw = pull_or_load_data()

    # Stage 1 — re-emit diagnostics if CSVs missing
    if not (os.path.exists(CONTRACTS_CSV_PATH)
            and os.path.exists(PAIRINGS_CSV_PATH)
            and os.path.exists(SPREADS_CSV_PATH)):
        stage1_diagnostics(raw)
    else:
        print("Stage 1 CSVs already present — skipping re-emit "
              "(delete to regenerate).")

    # Stage 2 — main test grid
    print("\n" + "=" * 72)
    print("STAGE 2 — TEST GRID")
    print("=" * 72)
    main_df = build_main_results(raw)
    main_df.to_csv(MAIN_RESULTS_PATH, index=False)
    print(f"\nMain results → {MAIN_RESULTS_PATH}  ({len(main_df)} rows)")

    # Stage 2 — long-horizon lagged regression
    print("\n" + "=" * 72)
    print("STAGE 2 — LAGGED REGRESSION (long horizon)")
    print("=" * 72)
    lagged_df = build_lagged_regression(raw)
    lagged_df.to_csv(LAGGED_RESULTS_PATH, index=False)
    print(f"\nLagged regression → {LAGGED_RESULTS_PATH}  ({len(lagged_df)} rows)")

    print("\nStage 2 complete. Both CSVs written. Awaiting review before stage 3.")


if __name__ == "__main__":
    main()
