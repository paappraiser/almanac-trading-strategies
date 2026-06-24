"""
Stock Trader's Almanac — Optimized Composite Scorer
=====================================================
Walk-forward optimized weights (2005-2026 OOS validation).
Uses consensus weights from grid-search optimization across SPY, QQQ, DIA.

Import this instead of the default AlmanacCompositeScorer for production use.
The optimized weights reduce Best Six Months MACD by 55% and increase
Presidential Election Cycle by 17% based on OOS performance.

Usage:
    from optimized_scorer import OptimizedAlmanacScorer
    scorer = OptimizedAlmanacScorer()
    score = scorer.composite_score(df)
    signal = scorer.signal(df)
"""

from stock_traders_almanac.rules import (
    AlmanacCompositeScorer, MonthlySeasonalitySignal,
    BestSixMonthsMACDSwitch, PresidentialElectionCycle,
    SeptemberOctoberEffect, TurnOfTheMonth, Super8Days,
    FirstTradingDay, PreHolidayEffect, SantaClausRally,
    MidMonthBulge, TaxLossJanuaryEffect,
)

# Consensus optimized weights (average across SPY/QQQ/DIA walk-forward)
OPTIMIZED_WEIGHTS = {
    "best_six_months_macd": 0.67,      # was 1.50 (-55%) — most over-weighted
    "monthly_seasonality": 1.03,        # was 1.00 (+3%)
    "presidential_cycle": 1.40,         # was 1.20 (+17%) — most under-weighted
    "september_october": 0.58,          # was 0.80 (-28%)
    "turn_of_month": 0.43,              # was 0.50 (-14%)
    "super_8": 0.61,                    # was 0.60 (+2%)
    "first_trading_day": 0.20,          # unchanged
    "pre_holiday": 0.30,                # unchanged
    "santa_claus_rally": 0.40,          # unchanged
    "mid_month_bulge": 0.40,            # unchanged
    "tax_loss_january": 0.30,           # unchanged
}

OPTIMIZED_THRESHOLD = 0.87  # was 1.00 — be less picky about going long


class OptimizedAlmanacScorer(AlmanacCompositeScorer):
    """
    AlmanacCompositeScorer with walk-forward optimized weights.
    
    Changes from default:
    - Best Six Months MACD: 1.50 -> 0.67 (over-weighted in default)
    - Presidential Cycle: 1.20 -> 1.40 (under-weighted in default)
    - September/October: 0.80 -> 0.58
    - Turn of Month: 0.50 -> 0.43
    - Threshold: 1.00 -> 0.87 (be less picky)
    
    OOS performance (2005-2026, 22 windows):
    - SPY: +6.5% ann, MaxDD -11.3% vs B&H -14.4%
    - QQQ: +11.0% ann, MaxDD -12.1% vs B&H -16.7%
    - DIA: +7.2% ann, MaxDD -10.1% vs B&H -13.7%
    """

    def __init__(self, threshold: float = None):
        super().__init__(weights=OPTIMIZED_WEIGHTS)
        self._optimized_threshold = threshold or OPTIMIZED_THRESHOLD

    def signal(self, df) -> "pd.Series":
        """Override signal with optimized threshold."""
        import pandas as pd
        comp = self.composite_score(df)
        signal = pd.Series(0, index=df.index)
        signal[comp > self._optimized_threshold] = 1
        signal[comp < -self._optimized_threshold] = -1
        return signal

    def regime(self, df) -> "pd.Series":
        """Overridden with tighter bands for optimized weights."""
        import pandas as pd
        comp = self.composite_score(df)
        t = self._optimized_threshold
        regime = pd.Series("neutral", index=df.index)
        regime[comp > 2.5] = "strong_bull"
        regime[(comp > t) & (comp <= 2.5)] = "bull"
        regime[(comp >= -t) & (comp <= t)] = "neutral"
        regime[(comp >= -2.5) & (comp < -t)] = "bear"
        regime[comp < -2.5] = "strong_bear"
        return regime


# Backward-compatible default scorer with recommended weights
recommended_scorer = OptimizedAlmanacScorer()
