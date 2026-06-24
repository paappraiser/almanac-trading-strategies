# Almanac Trading Strategies

Stock Trader's Almanac seasonal trading rule sets — fully codified, backtest-validated.

## What's Here

**14 quantitative rule sets** extracted from the Stock Trader's Almanac (Yale Hirsch / Jeffrey Hirsch), covering 70+ years of documented seasonal patterns since 1950.

### Flagship Strategies

| ID | Strategy | Description |
|----|----------|-------------|
| s700 | Best Six Months (MACD) | Long DJIA/SPY Nov-Apr, cash May-Oct with MACD timing |
| s701 | NASDAQ Best Eight Months | Tech sector's longer favorable season (Nov-Jun) |
| s706 | Almanac Composite | Weighted ensemble of all 11 seasonal rule sets |

### Calendar Patterns

| ID | Strategy | Historical Edge |
|----|----------|----------------|
| s702 | January Barometer Trifecta | All 3 positive → year up 90%, S&P avg +17.5% |
| s703 | Presidential Election Cycle | Pre-election year strongest (+17.2% S&P) |
| s704 | Super 8 Days | First 2 + last 3 + mid-month days capture most gains |
| s705 | Turn of the Month | Last day + first 4 days — monthly 401(k) bulge |
| s707 | September/October Effect | Sept worst month (-0.7% S&P), Oct "Bear Killer" |
| s708 | Santa Claus Rally | Last 5 Dec + first 2 Jan, S&P avg +1.3% |
| s709 | Pre-Holiday Effect | Day before holidays, 10x normal returns |
| s710 | First Trading Day | DJIA positive 60% of time |
| s711 | Monthly Seasonality | Ranked months since 1950 |
| s712 | Mid-Month Bulge | Trading days 9-15 institutional flows |
| s713 | Tax-Loss / January Effect | Dec weakness, Jan small-cap bounce |

## Backtest Results (2005-2024)

Composite seasonal signal separates good from bad periods across all major indices:

| Index | Bull Days (ann) | Bear Days (ann) | B&H (ann) |
|-------|----------------|-----------------|-----------|
| DIA   | **+14.5%**     | -6.3%           | +7.1%     |
| SPY   | **+16.2%**     | -9.4%           | +8.3%     |
| QQQ   | **+20.5%**     | -6.3%           | +13.7%    |

Bull days capture virtually all net market gains. Bear/neutral days collectively deliver negative returns.

## Quick Start

```python
from stock_traders_almanac import AlmanacCompositeScorer, BestSixMonthsMACDSwitch
import yfinance as yf

# Load data
df = yf.download('SPY', start='2015-01-01')

# Composite seasonal score (-5 to +5)
composite = AlmanacCompositeScorer()
score = composite.composite_score(df)       # continuous
signal = composite.signal(df)               # +1 bullish, -1 bearish, 0 neutral
regime = composite.regime(df)               # strong_bull/bull/neutral/bear/strong_bear

# Best Six Months standalone
best6 = BestSixMonthsMACDSwitch()
bsignal = best6.signal(df)                  # +1 long Nov-Apr, -1 cash May-Oct

# Generate a seasonal dashboard
from stock_traders_almanac import almanac_dashboard
print(almanac_dashboard(df, 2024))
```

## Use Cases

- **Regime filter**: Remove trades during low-score seasonal periods
- **Position sizing**: Scale up during Nov-Apr, scale down May-Oct
- **Meta-labeling**: Layer seasonal score on top of any existing strategy
- **Standalone timing**: Best Six Months MACD switch as a complete strategy
- **Dashboard**: Annual seasonal outlook for any index

## Installation

```bash
pip install pandas numpy yfinance
git clone https://github.com/paappraiser/almanac-trading-strategies.git
```

## Sources

- Stock Trader's Almanac (Yale Hirsch, est. 1967)
- Jeffrey A. Hirsch — current editor
- All statistics sourced from Almanac publications (since 1950 data)
- MACD parameters: standard 12-26-9 (as used by Almanac Tactical Switching Strategy)
