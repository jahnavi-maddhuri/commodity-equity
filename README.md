# Commodity-Equity Statistical Arbitrage

A quantitative trading strategy that exploits information diffusion lags between commodity futures prices and the equities of companies that depend on those commodities as primary inputs. When a commodity price moves significantly but the dependent equity has not yet repriced, a mean-reversion trade is entered on the equity leg only.

---

## Strategy Overview

**Core hypothesis:** Commodity price signals are publicly available, but cross-market information processing is slow. When input costs rise or fall sharply, equity markets take days to weeks to fully reflect the impact on a company's margins. This lag creates a statistically exploitable mispricing window.

**Type:** Statistical Arbitrage — Cross-Asset Mean Reversion  
**Instruments:** U.S.-listed equities (equity leg only; no direct commodity position)  
**Signal:** Z-score of the spread between implied equity move (from commodity) and actual equity move  
**Entry:** |Z-score| > k_sigma threshold  
**Exit:** Z-score reverts to ±0.25σ, or maximum holding period elapsed  
**Holding period:** ~10–16 days average  
**Trading frequency:** ~6–8 trades per pair per year

---

## Validated Pairs

Both pairs passed Engle-Granger cointegration (p < 0.05) and Granger causality tests on 2015–2025 daily data.

| Pair | Commodity | Ticker | Equity | Ticker | Coint p-value | Granger Lag | Status |
|---|---|---|---|---|---|---|---|
| Copper → Rockwell Automation | Copper Futures | HG=F | Rockwell Automation | ROK | 0.020 | 8 days | ✅ Active |
| Copper → TE Connectivity | Copper Futures | HG=F | TE Connectivity | TEL | 0.001 | 5 days | ⚠️ Under review |

> **Note:** Copper → ROK is the primary tradeable pair. Copper → TEL passes cointegration but shows negative Sharpe across most parameter combinations. It is retained for monitoring and further research.

---

## Key Results (Copper → ROK)

Optimized parameters: `signal_window = 15 days`, `k_sigma = 1.5σ`

| Test | Period | CAGR | Sharpe | Max Drawdown | Win Rate | Profit Factor |
|---|---|---|---|---|---|---|
| Training | 2015–2022 | 5.51% | 0.166 | -24.55% | 49.06% | 1.717 |
| Out-of-Sample | 2022–2025 | 2.58% | -0.005 | -21.60% | 45.00% | 1.281 |
| Trade War Scenario | 2018–2019 | 8.25% | 0.362 | -13.73% | 53.85% | 2.210 |
| COVID Scenario | 2020–2021 | -5.31% | -0.684 | -16.47% | 28.57% | 0.159 |
| Rate Hike Scenario | 2022 | 16.26% | 0.638 | -18.28% | 50.00% | 2.116 |

The strategy performs best in commodity-driven macroeconomic stress environments (trade wars, rate hike cycles) and breaks down during extreme systemic dislocations (COVID). A VIX regime filter is planned for Phase 2.

---

## Repository Structure

```
commodity-equity/
│
├── test_pairs.py                          # Part 1: Pair discovery and statistical validation
├── backtest.py                            # Part 2: Parameter optimization and backtesting
│
├── pair_test_results_1.csv                # Pair validation results — round 1 (ETF proxies)
├── pair_test_results_2.csv                # Pair validation results — round 2 (continuous futures)
├── pair_test_results_3.csv                # Pair validation results — round 3 (weekly beta)
├── pair_test_results_4.csv                # Pair validation results — round 4 (copper industrials)
├── pair_test_results_5.csv                # Pair validation results — round 5 (lumber + additional)
│
├── backtest_fixed_results.json            # Full backtest results for validated pairs
├── grid_fixed_Copper_to_ROK.csv           # Grid search results — Copper → ROK
├── grid_fixed_Copper_to_TEL.csv           # Grid search results — Copper → TEL
│
├── chart_*.png                            # Pair diagnostic charts (normalized prices,
│                                          #   rolling beta, spread, Z-score)
├── equity_fixed_*.png                     # Equity curves — train/test split with Z-score
├── grid_fixed_*.png                       # Sharpe ratio heatmaps from grid search
└── capital_allocation.png                 # Capital allocation chart across pairs
```

---

## Scripts

### `test_pairs.py` — Pair Discovery and Statistical Validation

Runs the full pair validation pipeline for any commodity-equity combination. Tests whether a pair has a statistically reliable mean-reverting relationship and whether the commodity leads the equity directionally.

**What it does:**
- Downloads daily price data via `yfinance` for commodity futures and equity tickers
- Runs ADF stationarity test on each series
- Runs **Engle-Granger cointegration test** — pair must have p-value < 0.05 to proceed
- Runs **Granger causality test** — confirms commodity leads equity, not the other way around
- Estimates rolling OLS beta (weekly returns, 252-day window)
- Computes spread and rolling Z-score
- Generates four-panel diagnostic chart per pair
- Outputs a ranked scorecard CSV

**To add a new pair**, edit the `PAIRS` list at the top of the file:
```python
PAIRS = [
    ("HG=F", "ROK", "Copper → Rockwell Automation"),
    ("HG=F", "TEL", "Copper → TE Connectivity"),
    # Add your pair here:
    ("HG=F", "HUBB", "Copper → Hubbell"),
]
```

**Outputs:**
- `pair_test_results.csv` — scorecard for all tested pairs
- `chart_*.png` — diagnostic chart per pair

---

### `backtest_fixed.py` — Parameter Optimization and Backtesting

Runs the full backtesting pipeline on validated pairs. Includes grid search over parameter combinations, walk-forward validation, scenario testing, and capital allocation.

**What it does:**
- Downloads data and computes rolling beta using weekly returns
- Splits data 70% training / 30% out-of-sample
- **Grid search** over all combinations of `signal_window` (5–90 days) and `k_sigma` (0.6–2.5σ), where k_sigma scales inversely with window length per professor guidance
- Selects best parameters by Sharpe ratio on training set
- Runs out-of-sample backtest with best parameters
- Runs **Back Test 2** (nearest 6 months) with best parameters
- Runs **scenario tests** across: US-China Trade War (2018–2019), COVID (2020–2021), 2022 Rate Hike Cycle
- Computes all performance metrics: Sharpe, Calmar, Max Drawdown, CAGR, Simple Return, Trading Frequency, Avg Hold, Win Rate, Profit Factor
- Computes **capital allocation** across pairs using equal, volatility-weighted, or Kelly method
- Uses fixed position sizing (95% of initial capital per trade) — no compounding between trades

**Key design decisions:**
- No stop-loss in Phase 1 (per professor guidance — add after parameter optimization)
- Weekly returns used for beta estimation; daily prices used for signal computation
- Position size is fixed fraction of *initial* capital, not current portfolio value

**To run:**
```bash
python backtest_fixed.py
```

**Outputs:**
- `backtest_fixed_results.json` — full structured results for all pairs and scenarios
- `equity_fixed_*.png` — portfolio equity curve with train/test split and Z-score panel
- `grid_fixed_*.png` — Sharpe ratio heatmap across all parameter combinations
- `grid_fixed_*.csv` — ranked grid search results (top 20 combinations per pair)
- `capital_allocation.png` — bar chart of capital allocation across pairs

---

## Installation

```bash
git clone https://github.com/jahnavi-maddhuri/commodity-equity.git
cd commodity-equity
pip install yfinance statsmodels pandas numpy matplotlib seaborn scipy
```

**Python version:** 3.9+  
**statsmodels version:** 0.14.0+ required (earlier versions have a compatibility issue with Python 3.12)

```bash
pip install --upgrade statsmodels
```

---

## How to Run

**Step 1 — Validate pairs:**
```bash
python test_pairs.py
```
Review `pair_test_results.csv`. Only proceed with pairs where both `Coint Pass = ✓` and `Granger Pass = ✓`.

**Step 2 — Run backtesting on validated pairs:**
```bash
python backtest.py
```
Review `backtest_fixed_results.json` and the generated charts. Key metrics to evaluate: OOS Sharpe > 0, OOS Profit Factor > 1, coherent green region in heatmap (not isolated hot spots).

---

## Parameter Reference

| Parameter | Default | Range Tested | Description |
|---|---|---|---|
| `signal_window` | 15 days | 5–90 days | Lookback for measuring commodity and equity returns |
| `k_sigma` | 1.5σ (ROK), 1.2σ (TEL) | 0.6–2.5σ | Entry threshold; scales inversely with window length |
| `beta_window` | 252 days | — | Rolling OLS window for beta estimation (weekly returns) |
| `zscore_window` | 60 days | — | Rolling window for Z-score normalization |
| `exit_threshold` | 0.25σ | — | Z-score level at which position is closed |
| `max_hold_days` | 20 days | 10–30 days | Maximum holding period before forced exit |
| `position_fraction` | 0.95 | — | Fraction of fixed initial capital deployed per trade |
| `commission` | 0.10% | — | Round-trip transaction cost per trade |

---

## Pair Selection Criteria

A pair must satisfy all of the following before being traded:

1. Commodity represents ≥ 20% of the company's COGS or operating expenses
2. Company is a relatively pure-play business (not a large conglomerate)
3. Engle-Granger cointegration p-value < 0.05
4. Granger causality p-value < 0.05 (commodity must lead equity)
5. Commodity futures have at least 5 years of clean daily price history
6. Equity is liquid enough to short without excessive borrowing costs

---

## Pairs Tested

19 pairs were tested across 5 commodity categories. Summary:

| Category | Pairs Tested | Passed Both Tests |
|---|---|---|
| Copper → Industrials | 9 | 2 (ROK, TEL) |
| Crude Oil → Airlines | 3 | 0 |
| Natural Gas → Fertilizer | 2 | 0 |
| Steel → Autos | 2 | 0 |
| Lumber → Homebuilders | 3 | 0 |

Full results in `pair_test_results_*.csv`.

---

## Roadmap

**Phase 1 (complete):**
- [x] Pair discovery framework
- [x] Statistical validation (cointegration + Granger causality)
- [x] Grid search parameter optimization
- [x] Walk-forward backtesting
- [x] Scenario analysis (Trade War, COVID, Rate Hike Cycle)
- [x] Capital allocation across pairs

**Phase 2 (planned):**
- [ ] VIX regime filter (skip entries when VIX > 30)
- [ ] Position-level stop-loss (-2% NAV)
- [ ] Expand pair universe (Copper → Hubbell, Copper → Freeport-McMoRan)
- [ ] Formal capacity test with market impact model
- [ ] Paper trading implementation

**Phase 3 (future):**
- [ ] Live trading integration
- [ ] Stop-gain optimization
- [ ] Extension to international equities

---

## Academic Context

This strategy exploits a deviation from the **semi-strong form of the Efficient Market Hypothesis**, which holds that all publicly available information is already priced in. The commodity-equity relationship requires cross-market analysis that most equity investors do not perform in real time, creating a bounded and persistent information diffusion lag.

Statistical foundations:
- **Engle-Granger cointegration** — confirms the spread is stationary and mean-reverting
- **Granger causality** — confirms commodity prices lead equity prices directionally
- **OLS regression** — estimates the commodity-beta (sensitivity) of each equity
- **Rolling Z-score** — normalizes the spread signal for entry/exit decisions
