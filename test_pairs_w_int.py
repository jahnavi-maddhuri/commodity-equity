"""
Producer-Intermediate Pair Testing
==================================
Tests whether replacing the commodity-futures leg with a pure-play
producer equity strengthens cointegration with consumer-equity targets.
This addresses path 2 of the professor's feedback: lumber → WFG → LEN.

All statistical methodology is identical to test_pairs.py — same date
window (2017-01-01 → 2020-03-01), same Granger maxlag (10), same beta
estimation (weekly-resampled). The only change is the first leg of each
pair: a producer equity replaces the commodity futures.

Internal helper functions (imported from test_pairs.py) treat the first
leg as 'commodity' and the second as 'equity'. That convention is kept
unchanged inside the helpers; all user-facing output (prints, CSV
columns, chart filenames) uses producer / consumer.

Run: python test_pairs_w_int.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from test_pairs import (
    download_pair,
    test_stationarity,
    test_cointegration,
    test_granger_causality,
    estimate_beta,
    compute_spread_and_zscore,
    plot_pair_analysis,        # panel titles say "Commodity" — reused unmodified
)


# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────

# (producer_ticker, consumer_ticker, label)
PAIRS = [
    ("FCX", "ROK", "Copper Producer (FCX) → Rockwell Automation"),
    ("FCX", "TEL", "Copper Producer (FCX) → TE Connectivity"),
    ("FCX", "AME", "Copper Producer (FCX) → Ametek"),
    ("FCX", "PH",  "Copper Producer (FCX) → Parker Hannifin"),
    ("FCX", "ETN", "Copper Producer (FCX) → Eaton Corp"),
    ("FCX", "IR",  "Copper Producer (FCX) → Ingersoll Rand"),
    ("FCX", "EMR", "Copper Producer (FCX) → Emerson Electric"),
    ("EOG", "AAL", "Oil Producer (EOG) → American Airlines"),
    ("EOG", "UAL", "Oil Producer (EOG) → United Airlines"),
    ("EOG", "DAL", "Oil Producer (EOG) → Delta Airlines"),
    ("EQT", "CF",  "Gas Producer (EQT) → CF Industries"),
    ("EQT", "NTR", "Gas Producer (EQT) → Nutrien"),
    ("NUE", "F",   "Steel Producer (NUE) → Ford"),
    ("NUE", "GM",  "Steel Producer (NUE) → General Motors"),
    ("WFG", "LEN", "Lumber Producer (WFG) → Lennar"),
    ("WFG", "DHI", "Lumber Producer (WFG) → D.R. Horton"),
    ("WFG", "PHM", "Lumber Producer (WFG) → PulteGroup"),
]

START_DATE = "2017-01-01"
END_DATE   = "2020-03-01"

COINT_PVALUE   = 0.05
GRANGER_PVALUE = 0.05

# Per-ticker data caveats — empty string for tickers without known issues.
DATA_CAVEATS = {
    "WFG": "WFG pre-2021 reconstructed series (NYSE listing began Feb 2021)",
    "NTR": "NTR exists only from 2018-01-02 (post-merger)",
    "IR":  "IR truncated start (2017-05-12), structural break Feb 29 2020",
}


def caveat_for(producer, consumer):
    parts = []
    for t in (producer, consumer):
        c = DATA_CAVEATS.get(t, "")
        if c and c not in parts:
            parts.append(c)
    return "; ".join(parts)


# ─────────────────────────────────────────────
# 2. MAIN RUNNER
# ─────────────────────────────────────────────
def run_all_pairs():
    results_summary = []

    print("=" * 60)
    print("PRODUCER-INTERMEDIATE PAIR TESTING PIPELINE")
    print(f"Period: {START_DATE} → {END_DATE}")
    print("=" * 60)

    for producer_ticker, consumer_ticker, label in PAIRS:
        print(f"\n{'─' * 60}")
        print(f"PAIR: {label}")
        print(f"Producer: {producer_ticker}  |  Consumer: {consumer_ticker}")
        caveat = caveat_for(producer_ticker, consumer_ticker)
        if caveat:
            print(f"Data caveat: {caveat}")
        print(f"{'─' * 60}")

        # ── Download data (producer goes into 'commodity' slot internally) ──
        try:
            df = download_pair(producer_ticker, consumer_ticker,
                               START_DATE, END_DATE)
        except Exception as e:
            print(f"  DATA ERROR: {e}")
            continue

        if len(df) < 500:
            print(f"  SKIPPED — insufficient data ({len(df)} rows)")
            continue

        print(f"  Data: {len(df)} trading days  "
              f"({df.index[0].date()} → {df.index[-1].date()})")

        # ── Stationarity (informational) ──
        adf_p = test_stationarity(df["commodity"], producer_ticker)
        adf_c = test_stationarity(df["equity"],    consumer_ticker)
        print(f"  ADF — {producer_ticker}: p={adf_p['p_value']} "
              f"({'stationary' if adf_p['stationary'] else 'non-stationary ✓'})")
        print(f"  ADF — {consumer_ticker}: p={adf_c['p_value']} "
              f"({'stationary' if adf_c['stationary'] else 'non-stationary ✓'})")

        # ── Cointegration ──
        coint_result = test_cointegration(df)
        print(f"  {coint_result['test']}: "
              f"p={coint_result['p_value']} → {coint_result['conclusion']}")

        # ── Granger causality (producer → consumer) ──
        granger_result = test_granger_causality(df)
        print(f"  {granger_result['test']}: "
              f"best p={granger_result['best_pvalue']} "
              f"at lag {granger_result['best_lag']}d "
              f"→ {granger_result['conclusion']}")

        # ── Beta estimation (weekly-resampled, kept unchanged) ──
        beta_info = estimate_beta(df)
        print(f"  Beta (β): {beta_info['beta']}  |  "
              f"R²: {beta_info['r_squared']}  |  "
              f"Alpha: {beta_info['alpha']}")

        # ── Spread and Z-score ──
        spread_df = compute_spread_and_zscore(df, beta_info["beta"])

        # ── Plot (panel titles say "Commodity" — that's from the reused helper) ──
        chart_path = f"chart_intermediate_{producer_ticker}_{consumer_ticker}.png"
        plot_pair_analysis(
            df, spread_df, beta_info,
            coint_result, granger_result,
            label, save_path=chart_path,
        )

        # ── Score the pair ──
        both_passed = coint_result["passed"] and granger_result["passed"]
        score = sum([
            coint_result["passed"],
            granger_result["passed"],
            beta_info["r_squared"] > 0.1,
            beta_info["beta"] < 0,
        ])

        results_summary.append({
            "Pair":               label,
            "Producer":           producer_ticker,
            "Consumer":           consumer_ticker,
            "Data Points":        len(df),
            "Coint p-value":      coint_result["p_value"],
            "Coint Pass":         "✓" if coint_result["passed"] else "✗",
            "Granger p-value":    granger_result["best_pvalue"],
            "Granger Lag (days)": granger_result["best_lag"],
            "Granger Pass":       "✓" if granger_result["passed"] else "✗",
            "Beta (β)":           beta_info["beta"],
            "R²":                 beta_info["r_squared"],
            "Both Tests Pass":    "✓✓ TRADE" if both_passed else "✗  SKIP",
            "Score":              f"{score}/4",
            "data_caveats":       caveat,
        })

    # ── Scorecard ──
    print(f"\n{'=' * 60}")
    print("PRODUCER-INTERMEDIATE PAIR SCORECARD")
    print(f"{'=' * 60}")

    summary_df = pd.DataFrame(results_summary)

    if summary_df.empty:
        print("No pairs successfully tested.")
        return summary_df

    summary_df["_sort"] = summary_df["Both Tests Pass"].apply(
        lambda x: 0 if "TRADE" in x else 1
    )
    summary_df = summary_df.sort_values(["_sort", "Granger p-value"])
    summary_df = summary_df.drop(columns=["_sort"])

    print_cols = ["Pair", "Coint Pass", "Coint p-value",
                  "Granger Pass", "Granger Lag (days)",
                  "Beta (β)", "R²", "Both Tests Pass"]
    print(summary_df[print_cols].to_string(index=False))

    summary_df.to_csv("pair_test_results_intermediate.csv", index=False)
    print("\nFull results saved → pair_test_results_intermediate.csv")
    print("Charts saved as chart_intermediate_*.png")

    # ── End-of-run summary ──
    n_total   = len(summary_df)
    n_coint   = (summary_df["Coint Pass"]   == "✓").sum()
    n_granger = (summary_df["Granger Pass"] == "✓").sum()
    n_both    = (summary_df["Both Tests Pass"].str.contains("TRADE")).sum()
    print(f"\nSummary: {n_coint}/{n_total} passed cointegration, "
          f"{n_granger}/{n_total} passed Granger, "
          f"{n_both}/{n_total} passed both.")

    return summary_df


# ─────────────────────────────────────────────
# 3. ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    results = run_all_pairs()
