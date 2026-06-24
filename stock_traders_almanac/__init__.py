# Almanac Trading Strategies
__version__ = "1.0.0"

from .rules import (
    # Utility
    _normalize_df,
    macd,
    # Core rule sets
    MonthlySeasonalitySignal,
    BestSixMonthsMACDSwitch,
    NasdaqBestEightMonths,
    SantaClausRally,
    JanuaryBarometerTrifecta,
    TurnOfTheMonth,
    Super8Days,
    FirstTradingDay,
    PreHolidayEffect,
    PresidentialElectionCycle,
    SeptemberOctoberEffect,
    MidMonthBulge,
    TaxLossJanuaryEffect,
    # Composite
    AlmanacCompositeScorer,
    # Constants
    ALMANAC_RULE_SETS,
    BEST_SIX_MONTHS,
    WORST_SIX_MONTHS,
    BEST_EIGHT_MONTHS_NASDAQ,
    _MONTHLY_SEASONALITY_DJIA,
    _MONTHLY_SEASONALITY_SPX,
    # Functions
    generate_all_signals,
    almanac_dashboard,
)
