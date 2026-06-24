# Almanac Trading Strategies

14 quantitative rule sets from the Stock Trader's Almanac — codified, walk-forward validated, production-ready.

Covers 75 years of documented seasonal patterns (1950–2026) across DJIA, S&P 500, and NASDAQ. Every rule exposes `.score()` (continuous) and `.signal()` (binary) for direct strategy integration.

---

## Results at a Glance

### In-Sample Regime Split (2005–2026)

Composite seasonal signal separates good periods from bad across all three indices:

| Index | Bull days (ann) | Bear days (ann) | B&H (ann) |
|-------|:---------------:|:---------------:|:---------:|
| DIA   | **+14.5%**      | −6.3%           | +7.1%     |
| SPY   | **+16.2%**      | −9.4%           | +8.3%     |
| QQQ   | **+20.5%**      | −6.3%           | +13.7%    |

### Walk-Forward OOS (2005–2026, 22 windows, 5yr train / 1yr test, 5bps costs)

| Ticker | Optimized ann | Default ann | B&H ann | Opt Max DD | B&H Max DD |
|--------|:-----------:|:----------:|:------:|:----------:|:----------:|
| SPY    | +6.5%       | +6.4%      | +14.1% | **−11.3%** | −14.4%     |
| QQQ    | +11.0%      | +12.9%     | +20.7% | **−12.1%** | −16.7%     |
| DIA    | +7.2%       | +6.1%      | +12.7% | **−10.1%** | −13.7%     |

### Crisis Protection

| Year | Strategy | B&H | Delta |
|------|:--------:|:---:|:-----:|
| 2008 | −2.4% | −31.1% | **+28.7%** |
| 2018 | +0.4% | −3.2% | **+3.6%** |
| 2022 | −0.0% | −15.8% | **+15.8%** |

The strategy is a **risk-reduction overlay**, not a standalone alpha source. It consistently cuts drawdowns 15–35% at the cost of trailing in momentum bull markets. Best used as a regime filter or position sizer on top of other strategies.

---

## Quick Start

```bash
pip install pandas numpy yfinance
git clone https://github.com/paappraiser/almanac-trading-strategies.git
```

```python
from stock_traders_almanac import AlmanacCompositeScorer, BestSixMonthsMACDSwitch
from stock_traders_almanac.optimized_scorer import OptimizedAlmanacScorer
import yfinance as yf

df = yf.download("SPY", start="2015-01-01")

# Default weights (equal emphasis)
scorer = AlmanacCompositeScorer()
score  = scorer.composite_score(df)   # continuous −5 to +5
signal = scorer.signal(df)            # +1 bullish, −1 bearish, 0 neutral
regime = scorer.regime(df)            # strong_bull/bull/neutral/bear/strong_bear

# Optimized weights (walk-forward tuned)
opt = OptimizedAlmanacScorer()
opt_score  = opt.composite_score(df)
opt_signal = opt.signal(df)

# Best Six Months standalone
best6 = BestSixMonthsMACDSwitch()
b6_signal = best6.signal(df)          # +1 long Nov–Apr, −1 cash May–Oct

# Annual seasonal dashboard
from stock_traders_almanac import almanac_dashboard
print(almanac_dashboard(df, 2025))
```

---

## Rule Sets

### Flagship (s700–s706)

| ID | Strategy | Category | Key stat |
|----|----------|----------|----------|
| s700 | Best Six Months (MACD) | seasonal | DJIA Nov–Apr +7.3% vs May–Oct +0.8% since 1950 |
| s701 | NASDAQ Best Eight Months | seasonal | Nov–Jun, MACD-timed exit |
| s706 | Almanac Composite | ensemble | Weighted blend of all 11 rule sets |

### Calendar Patterns (s702–s713)

| ID | Strategy | Key stat |
|----|----------|----------|
| s702 | January Barometer Trifecta | All 3 positive → S&P up 90%, avg +17.5% |
| s703 | Presidential Election Cycle | Pre-election year avg +17.2%, midterm weakest |
| s704 | Super 8 Days | First 2 + last 3 + mid-month 9–11 capture most gains |
| s705 | Turn of the Month | Last day + first 4 — monthly 401(k) bulge |
| s707 | September/October Effect | Sept avg −0.7% (worst), Oct "Bear Killer" |
| s708 | Santa Claus Rally | Last 5 Dec + first 2 Jan, S&P avg +1.3% |
| s709 | Pre-Holiday Effect | Day before holidays, 10× normal returns |
| s710 | First Trading Day | DJIA positive 60% of time, avg +0.25% |
| s711 | Monthly Seasonality | Ranked months since 1950 (Apr best, Sept worst) |
| s712 | Mid-Month Bulge | Trading days 9–15, institutional flows |
| s713 | Tax-Loss / January Effect | Dec weakness, Jan small-cap bounce |

---

## Optimized Weights

Walk-forward optimization (grid search, 22 OOS windows) found the default weights over-emphasize Best Six Months and under-weight the Presidential Cycle:

| Signal | Default | Optimized | Change |
|--------|:-------:|:---------:|:------:|
| Best Six Months MACD | 1.50 | 0.67 | **−55%** |
| Presidential Cycle | 1.20 | 1.40 | **+17%** |
| September/October | 0.80 | 0.58 | −28% |
| Turn of Month | 0.50 | 0.43 | −14% |
| Monthly Seasonality | 1.00 | 1.03 | +3% |
| Super 8 | 0.60 | 0.61 | +2% |
| Short-term signals (5) | 0.20–0.40 | unchanged | — |
| **Threshold** | **1.00** | **0.87** | be less picky |

Use `OptimizedAlmanacScorer` for production. The default `AlmanacCompositeScorer` remains available as the "textbook Almanac" baseline.

---

## Use Cases

- **Regime filter** — skip trades when composite < −2.0
- **Position sizing** — 1.5× in strong_bull, 0.5× in strong_bear
- **Meta-labeling** — layer seasonal score onto any ML or technical strategy
- **Cron dashboard** — daily seasonal outlook via `almanac_dashboard()`
- **Standalone timing** — Best Six Months MACD switch as a complete strategy

---

## Sources

- Stock Trader's Almanac (Yale Hirsch, est. 1967)
- Jeffrey A. Hirsch — current editor, *Stock Trader's Almanac 2024–2026*
- All statistics from Almanac publications, DJIA/S&P since 1950
- MACD parameters: 12-26-9 (Almanac Tactical Switching Strategy standard)
- Walk-forward results: [`WALK_FORWARD_RESULTS.md`](WALK_FORWARD_RESULTS.md)
- Backtest script: `walk_forward_backtest.py`
