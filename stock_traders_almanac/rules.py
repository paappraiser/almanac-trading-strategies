"""
Stock Trader's Almanac — Quantitative Rule Sets
================================================
Codified rule sets from the Stock Trader's Almanac (Yale Hirsch / Jeffrey Hirsch).
All rules are based on patterns documented since 1950 for DJIA, S&P 500, NASDAQ.

Each class is a self-contained rule set with:
  - .score(df) → returns a signal score (-2 to +2)
  - .signal(df) → returns binary/ternary signal
  - .description() → human-readable explanation
  - Historical performance notes inline

Dependencies: pandas, numpy
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _trading_days_in_range(df: pd.DataFrame, start_mmdd: str, end_mmdd: str) -> pd.DatetimeIndex:
    """Get trading days between two MMDD dates within the dataframe index."""
    idx = df.index if hasattr(df, 'index') else df
    # Extract month-day strings
    mmdd = idx.strftime('%m%d')
    if start_mmdd <= end_mmdd:
        return idx[(mmdd >= start_mmdd) & (mmdd <= end_mmdd)]
    else:
        # Wrap around year (e.g. 1224 to 0105)
        return idx[(mmdd >= start_mmdd) | (mmdd <= end_mmdd)]


def _nth_trading_day(df: pd.DataFrame, n: int, month: int = None) -> pd.DatetimeIndex:
    """Get the nth trading day of each month (or specific month)."""
    idx = df.index
    result = []
    if month is not None:
        mask = (idx.month == month)
        groups = [(month, idx[mask])]
        for mo, group in groups:
            for year in pd.unique(group.year):
                yr_dates = group[group.year == year]
                if len(yr_dates) >= n:
                    result.append(yr_dates[n - 1])
    else:
        for year in pd.unique(idx.year):
            for mo in range(1, 13):
                yr_mo_dates = idx[(idx.year == year) & (idx.month == mo)]
                if len(yr_mo_dates) >= n:
                    result.append(yr_mo_dates[n - 1])
    return pd.DatetimeIndex(result)


def _last_n_trading_days(df: pd.DataFrame, n: int, month: int = 12) -> pd.DatetimeIndex:
    """Get the last n trading days of a specific month each year."""
    idx = df.index
    mask = (idx.month == month)
    month_idx = idx[mask]
    # Group by year and take last n
    result = []
    for year in pd.unique(month_idx.year):
        yr_dates = month_idx[month_idx.year == year]
        if len(yr_dates) >= n:
            result.append(yr_dates[-n:])
    if result:
        return pd.DatetimeIndex(np.concatenate([r.values for r in result]))
    return pd.DatetimeIndex([])


def _first_n_trading_days(df: pd.DataFrame, n: int, month: int = 1) -> pd.DatetimeIndex:
    """Get the first n trading days of a specific month each year."""
    idx = df.index
    mask = (idx.month == month)
    month_idx = idx[mask]
    result = []
    for year in pd.unique(month_idx.year):
        yr_dates = month_idx[month_idx.year == year]
        if len(yr_dates) >= n:
            result.append(yr_dates[:n])
    if result:
        return pd.DatetimeIndex(np.concatenate([r.values for r in result]))
    return pd.DatetimeIndex([])


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance MultiIndex columns to flat column names (Close, Open, etc.)."""
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance returns ('Close', 'DIA'), ('Open', 'DIA'), etc.
        # Drop the ticker level, keep only OHLCV
        flat_cols = {}
        for col_tuple in df.columns:
            col_name = col_tuple[0]  # 'Close', 'Open', etc.
            flat_cols[col_name] = df[col_tuple]
        return pd.DataFrame(flat_cols, index=df.index)
    return df


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Compute MACD indicator. Returns DataFrame with macd, signal, histogram."""
    close = pd.Series(close).squeeze()
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({'macd': macd_line, 'signal': signal_line, 'histogram': hist}, index=close.index)


def _is_pre_holiday(date: pd.Timestamp) -> bool:
    """Check if a date is a pre-holiday trading day (day before market holiday)."""
    # US market holidays
    year = date.year
    month = date.month
    day = date.day
    weekday = date.dayofweek  # 0=Mon, 4=Fri

    tomorrow = date + timedelta(days=1)
    tmo_month = tomorrow.month
    tmo_day = tomorrow.day
    tmo_weekday = tomorrow.dayofweek

    # Tomorrow is Saturday or Sunday → Friday is pre-weekend, not pre-holiday
    # Check major US holidays:

    # New Year's Day (Jan 1)
    if tmo_month == 1 and tmo_day == 1:
        return True

    # MLK Day (3rd Monday of January)
    if tmo_month == 1 and tmo_weekday == 0 and 15 <= tmo_day <= 21:
        return True

    # Presidents Day (3rd Monday of February)
    if tmo_month == 2 and tmo_weekday == 0 and 15 <= tmo_day <= 21:
        return True

    # Memorial Day (last Monday of May)
    if tmo_month == 5 and tmo_weekday == 0 and tmo_day >= 25:
        return True

    # Independence Day (Jul 4)
    if tmo_month == 7 and tmo_day == 4:
        return True
    # If Jul 4 is Saturday, observed Friday; if Sunday, observed Monday
    if month == 7 and day == 3 and weekday == 4 and tmo_weekday == 5:
        return True  # Friday before Saturday Jul 4

    # Labor Day (1st Monday of September)
    if tmo_month == 9 and tmo_weekday == 0 and 1 <= tmo_day <= 7:
        return True

    # Thanksgiving (4th Thursday of November)
    if tmo_month == 11 and tmo_weekday == 4 and 22 <= tmo_day <= 28:
        return True

    # Christmas (Dec 25)
    if tmo_month == 12 and tmo_day == 25:
        return True
    # If Dec 25 is Saturday, observed Friday; if Sunday, observed Monday
    if month == 12 and day == 24 and weekday == 4 and tmo_weekday == 5:
        return True

    return False


# ═══════════════════════════════════════════════════════════════════════════════
# MONTHLY SEASONALITY RULES
# ═══════════════════════════════════════════════════════════════════════════════

# DJIA monthly rankings & average returns since 1950 (from Stock Trader's Almanac)
# Rank 1 = best month, 12 = worst month
_MONTHLY_SEASONALITY_DJIA = {
    1:  {"rank": 4,  "avg_pct": 0.8,  "pct_positive": 60, "name": "January"},
    2:  {"rank": 10, "avg_pct": -0.1, "pct_positive": 52, "name": "February"},
    3:  {"rank": 5,  "avg_pct": 0.7,  "pct_positive": 58, "name": "March"},
    4:  {"rank": 1,  "avg_pct": 1.9,  "pct_positive": 63, "name": "April"},
    5:  {"rank": 8,  "avg_pct": 0.1,  "pct_positive": 56, "name": "May"},
    6:  {"rank": 11, "avg_pct": -0.2, "pct_positive": 50, "name": "June"},
    7:  {"rank": 3,  "avg_pct": 1.0,  "pct_positive": 57, "name": "July"},
    8:  {"rank": 12, "avg_pct": -0.3, "pct_positive": 52, "name": "August"},
    9:  {"rank": 12, "avg_pct": -0.6, "pct_positive": 45, "name": "September"},
    10: {"rank": 6,  "avg_pct": 0.5,  "pct_positive": 54, "name": "October"},
    11: {"rank": 2,  "avg_pct": 1.5,  "pct_positive": 65, "name": "November"},
    12: {"rank": 3,  "avg_pct": 1.4,  "pct_positive": 73, "name": "December"},
}

_MONTHLY_SEASONALITY_SPX = {
    1:  {"rank": 6,  "avg_pct": 0.7,  "pct_positive": 59, "name": "January"},
    2:  {"rank": 10, "avg_pct": 0.0,  "pct_positive": 53, "name": "February"},
    3:  {"rank": 5,  "avg_pct": 1.0,  "pct_positive": 60, "name": "March"},
    4:  {"rank": 3,  "avg_pct": 1.4,  "pct_positive": 68, "name": "April"},
    5:  {"rank": 8,  "avg_pct": 0.4,  "pct_positive": 60, "name": "May"},
    6:  {"rank": 9,  "avg_pct": 0.1,  "pct_positive": 53, "name": "June"},
    7:  {"rank": 4,  "avg_pct": 1.0,  "pct_positive": 57, "name": "July"},
    8:  {"rank": 11, "avg_pct": -0.1, "pct_positive": 56, "name": "August"},
    9:  {"rank": 12, "avg_pct": -0.7, "pct_positive": 42, "name": "September"},
    10: {"rank": 7,  "avg_pct": 0.8,  "pct_positive": 60, "name": "October"},
    11: {"rank": 1,  "avg_pct": 1.6,  "pct_positive": 68, "name": "November"},
    12: {"rank": 2,  "avg_pct": 1.5,  "pct_positive": 74, "name": "December"},
}

# Best Six Months vs Worst Six Months (DJIA since 1950)
# Nov-Apr avg +7.3%, May-Oct avg +0.8%
BEST_SIX_MONTHS = [11, 12, 1, 2, 3, 4]   # November through April
WORST_SIX_MONTHS = [5, 6, 7, 8, 9, 10]    # May through October
BEST_EIGHT_MONTHS_NASDAQ = [11, 12, 1, 2, 3, 4, 5, 6]  # NASDAQ Nov-Jun


class MonthlySeasonalitySignal:
    """Signal based on which month we're in and its historical rank."""

    def __init__(self, index: str = "DJIA"):
        self.seasonality = _MONTHLY_SEASONALITY_DJIA if index == "DJIA" else _MONTHLY_SEASONALITY_SPX
        self.index = index

    def score(self, date: pd.Timestamp) -> float:
        """Return -2 to +2 based on month rank."""
        m = date.month
        info = self.seasonality[m]
        rank = info["rank"]
        # Map rank 1-12 to score +2.0 to -2.0
        return 2.0 - (4.0 * (rank - 1) / 11)

    def signal(self, date: pd.Timestamp) -> int:
        """+1 if in top 4 months, -1 if in bottom 4, 0 otherwise."""
        m = date.month
        rank = self.seasonality[m]["rank"]
        if rank <= 4:
            return 1
        elif rank >= 9:
            return -1
        return 0

    def is_best_six(self, month: int) -> bool:
        return month in BEST_SIX_MONTHS

    def is_worst_six(self, month: int) -> bool:
        return month in WORST_SIX_MONTHS

    @staticmethod
    def description() -> str:
        return ("Monthly seasonality: April is best DJIA month (avg +1.9% since 1950), "
                "September worst (avg -0.6%). Nov-Dec-Jan strongest 3-month span (S&P avg +4.2%).")


# ═══════════════════════════════════════════════════════════════════════════════
# BEST SIX MONTHS — MACD-TIMED SEASONAL SWITCH
# ═══════════════════════════════════════════════════════════════════════════════

class BestSixMonthsMACDSwitch:
    """
    Stock Trader's Almanac Tactical Seasonal Switching Strategy.

    Core rule: Long DJIA/SPY from Nov 1 through Apr 30, switch to cash/fixed
    income May-October. Enhanced with MACD (12,26,9) timing for precise entry/exit.

    Entry window: Monitor MACD starting mid-October for buy signal
    Exit window: Monitor MACD starting mid-April for sell signal

    DJIA since 1950: Best Six Months avg +7.3% vs Worst Six Months avg +0.8%

    NASDAQ uses Best Eight Months (Nov-Jun).
    """

    def __init__(self, use_nasdaq: bool = False):
        self.use_nasdaq = use_nasdaq
        self.best_months = BEST_EIGHT_MONTHS_NASDAQ if use_nasdaq else BEST_SIX_MONTHS
        self.worst_months = WORST_SIX_MONTHS if not use_nasdaq else [7, 8, 9, 10]
        # Entry monitoring start: mid-October (or never for NASDAQ which doesn't exit)
        self.entry_monitor_start = 10  # October
        self.exit_monitor_start = 4    # April (or June for NASDAQ)
        if use_nasdaq:
            self.exit_monitor_start = 6  # June

    def score(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute seasonal MACD switch score for each date.
        Score ranges -2 (worst months, MACD sell confirmed) to +2 (best months, MACD buy confirmed).

        Logic:
        - In best months (Nov-Apr): score +1 base, +2 if MACD buy confirmed recently
        - In worst months (May-Oct): score -1 base, -2 if MACD sell confirmed recently
        - During transition windows: score based on MACD crossover direction
        """
        df = _normalize_df(df)
        close = df["Close"]
        m = macd(close)
        scores = pd.Series(0.0, index=df.index)

        for i in range(1, len(df)):
            date = df.index[i]
            month = date.month
            prev_close = close.iloc[i]

            if month in self.best_months:
                base = 1.0
                # Check if MACD is bullish (MACD > signal)
                if m["macd"].iloc[i] > m["signal"].iloc[i]:
                    base = 2.0
                # Check if MACD buy crossover recently (histogram turned positive)
                if (m["histogram"].iloc[i] > 0 and
                    m["histogram"].iloc[max(0, i - 5):i].min() <= 0):
                    base = 2.0
                scores.iloc[i] = base
            elif month in self.worst_months:
                base = -1.0
                if m["macd"].iloc[i] < m["signal"].iloc[i]:
                    base = -2.0
                if (m["histogram"].iloc[i] < 0 and
                    m["histogram"].iloc[max(0, i - 5):i].max() >= 0):
                    base = -2.0
                scores.iloc[i] = base
            else:
                scores.iloc[i] = 0.0

        return scores

    def signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Binary signal: long in best months with MACD confirmation, neutral/cash otherwise.
        +1 = long, 0 = cash, -1 = short/cash.
        Returns Series with same index as df.
        """
        scores = self.score(df)
        signal = pd.Series(0, index=df.index)
        signal[scores >= 1.0] = 1
        signal[scores <= -1.5] = -1  # Strong sell only
        return signal

    def get_macd_crossovers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return dates of MACD crossovers (buy/sell signals)."""
        df = _normalize_df(df)
        close = df["Close"]
        m = macd(close)
        crossovers = []
        for i in range(1, len(m)):
            if m["histogram"].iloc[i - 1] <= 0 and m["histogram"].iloc[i] > 0:
                crossovers.append((df.index[i], "BUY"))
            elif m["histogram"].iloc[i - 1] >= 0 and m["histogram"].iloc[i] < 0:
                crossovers.append((df.index[i], "SELL"))
        return pd.DataFrame(crossovers, columns=["date", "signal"])

    @staticmethod
    def description() -> str:
        return (
            "Best Six Months Switching Strategy (Almanac flagship): "
            "Long DJIA/SPY Nov 1 - Apr 30, cash May-Oct. MACD (12,26,9) "
            "times precise entry/exit. Since 1950: Best Months avg +7.3% "
            "vs Worst Months avg +0.8% for DJIA."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SANTA CLAUS RALLY
# ═══════════════════════════════════════════════════════════════════════════════

class SantaClausRally:
    """
    Santa Claus Rally (SCR)
    Defined as the last 5 trading days of December + first 2 trading days of January.
    7-day window, S&P 500 averages +1.3% since 1950, positive ~76% of the time.

    This is a SHORT-TERM seasonal pattern — useful for tactical positioning
    rather than multi-month allocation.
    """

    def __init__(self, threshold_pct: float = 0.5):
        self.threshold = threshold_pct  # minimum % to consider rally "confirmed"

    def is_in_window(self, date: pd.Timestamp) -> bool:
        """Check if date falls in the SCR window (last 5 Dec + first 2 Jan)."""
        month = date.month
        day = date.day
        # Approximate: last 5 business days of Dec = Dec 24-31
        if month == 12 and day >= 24:
            return True
        # First 2 business days of Jan = Jan 1-5
        if month == 1 and day <= 5:
            return True
        return False

    def score(self, df: pd.DataFrame) -> pd.Series:
        """Score: +2 inside SCR window (bullish bias), 0 outside. Historical avg +1.3%."""
        scores = pd.Series(0.0, index=df.index)
        for i in range(len(df)):
            if self.is_in_window(df.index[i]):
                scores.iloc[i] = 2.0
        return scores

    def signal(self, df: pd.DataFrame) -> pd.Series:
        """Binary: +1 inside SCR window, 0 otherwise."""
        signal = pd.Series(0, index=df.index)
        for i in range(len(df)):
            if self.is_in_window(df.index[i]):
                signal.iloc[i] = 1
        return signal

    def check_actual_rally(self, df: pd.DataFrame, year: int) -> Tuple[float, bool]:
        """Calculate the actual SCR return for a given year. Returns (return_pct, was_positive)."""
        df = _normalize_df(df)
        dec_last5 = _last_n_trading_days(df, 5, 12)
        jan_first2 = _first_n_trading_days(df, 2, 1)
        # Need next year's Jan for this year's SCR
        dec_dates = dec_last5[dec_last5.year == year]
        jan_dates = jan_first2[jan_first2.year == year + 1]

        if len(dec_dates) == 0 or len(jan_dates) == 0:
            return 0.0, False

        combined = pd.DatetimeIndex(list(dec_dates) + list(jan_dates)).sort_values()
        if len(combined) < 2:
            return 0.0, False

        start_price = df.loc[combined[0], "Close"]
        end_price = df.loc[combined[-1], "Close"]
        ret = (end_price / start_price - 1) * 100
        return ret, ret > self.threshold

    @staticmethod
    def description() -> str:
        return (
            "Santa Claus Rally: Last 5 trading days Dec + first 2 trading days Jan. "
            "S&P avg +1.3% since 1950, positive 76% of time. Yale Hirsch 1972. "
            "Part of January Indicator Trifecta."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# JANUARY BAROMETER TRIFECTA
# ═══════════════════════════════════════════════════════════════════════════════

class JanuaryBarometerTrifecta:
    """
    Three sequential January indicators (Stock Trader's Almanac):

    1. Santa Claus Rally (SCR) — last 5 Dec + first 2 Jan
    2. First Five Days — first 5 trading days of January ("Early Warning System")
    3. January Barometer — full January performance

    Combined (Trifecta):
    - All 3 positive → full year up ~90% of time, avg +17.5% S&P since 1950
    - All 3 negative → flat/negative year (only 1 exception: 1982)
    - Mixed → less predictive
    """

    def __init__(self):
        self.scr = SantaClausRally()

    def calculate_for_year(self, df: pd.DataFrame, year: int) -> Dict:
        """
        Calculate all three January indicators for a given year.
        Returns dict with SCR, First5Days, JanBarometer results.
        """
        df = _normalize_df(df)
        result = {
            "year": year,
            "SCR": None,
            "SCR_pct": 0.0,
            "First5Days": None,
            "First5Days_pct": 0.0,
            "JanBarometer": None,
            "JanBarometer_pct": 0.0,
            "Trifecta_positive": False,
            "Trifecta_negative": False,
        }

        # Santa Claus Rally (spans Dec year-1 to Jan year)
        if year > df.index[0].year:
            scr_ret, scr_pos = self.scr.check_actual_rally(df, year - 1)
            result["SCR"] = scr_pos
            result["SCR_pct"] = scr_ret

        # First Five Days of January
        jan_first5 = _first_n_trading_days(df, 5, 1)
        jan_first5_yr = jan_first5[jan_first5.year == year]
        if len(jan_first5_yr) >= 2:
            start_px = df.loc[jan_first5_yr[0], "Close"]
            end_px = df.loc[jan_first5_yr[-1], "Close"]
            ret = (end_px / start_px - 1) * 100
            result["First5Days"] = ret > 0
            result["First5Days_pct"] = ret

        # Full January
        jan_all = df[(df.index.year == year) & (df.index.month == 1)]
        if len(jan_all) >= 2:
            start_px = jan_all["Close"].iloc[0]
            end_px = jan_all["Close"].iloc[-1]
            ret = (end_px / start_px - 1) * 100
            result["JanBarometer"] = ret > 0
            result["JanBarometer_pct"] = ret

        # Trifecta check
        if result["SCR"] is not None and result["First5Days"] is not None and result["JanBarometer"] is not None:
            if result["SCR"] and result["First5Days"] and result["JanBarometer"]:
                result["Trifecta_positive"] = True
            if not result["SCR"] and not result["First5Days"] and not result["JanBarometer"]:
                result["Trifecta_negative"] = True

        return result

    def score(self, df: pd.DataFrame) -> pd.Series:
        """
        Score based on January Trifecta.
        After January data is in, score = +2 if all three positive, -2 if all three negative.
        Before January, score = 0 (unknown).
        Score persists through the year once determined.
        """
        df = _normalize_df(df)
        scores = pd.Series(0.0, index=df.index)
        current_trifecta_score = 0.0

        for year in range(df.index[0].year, df.index[-1].year + 1):
            result = self.calculate_for_year(df, year)
            if result["Trifecta_positive"]:
                current_trifecta_score = 2.0
            elif result["Trifecta_negative"]:
                current_trifecta_score = -2.0
            else:
                current_trifecta_score = 0.0

            # Apply score from February onward (after January data is known)
            yr_mask = (df.index.year == year) & (df.index.month >= 2)
            scores.loc[yr_mask] = current_trifecta_score

        return scores

    def signal(self, df: pd.DataFrame) -> pd.Series:
        """+1 if Trifecta positive, -1 if Trifecta negative, 0 otherwise."""
        scores = self.score(df)
        signal = pd.Series(0, index=df.index)
        signal[scores >= 1.5] = 1
        signal[scores <= -1.5] = -1
        return signal

    @staticmethod
    def description() -> str:
        return (
            "January Barometer Trifecta: SCR + First 5 Days + Full January. "
            "All 3 positive → year up 90% of time, S&P avg +17.5% since 1950. "
            "All 3 negative → flat/negative year (one exception: 1982)."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TURN OF THE MONTH EFFECT (Monthly 5-Day Bulge)
# ═══════════════════════════════════════════════════════════════════════════════

class TurnOfTheMonth:
    """
    Turn of the Month Effect (Monthly Bulge)
    - Last trading day of month + first 4 trading days of next month
    - One of the most persistent calendar effects
    - Disproportionate share of monthly gains occur in these 5 days
    - Tied to institutional flows: month-end rebalancing, payday 401(k) flows
    """

    def __init__(self):
        pass

    def is_in_window(self, date: pd.Timestamp, df: pd.DataFrame) -> bool:
        """Check if date is in turn-of-the-month window."""
        # Last trading day of current month
        month = date.month
        year = date.year
        # Get all dates in this month
        month_dates = df.index[(df.index.year == year) & (df.index.month == month)]
        if len(month_dates) == 0:
            return False
        last_trading_day = month_dates[-1]

        # First 4 trading days of next month
        next_month = month + 1 if month < 12 else 1
        next_year = year if month < 12 else year + 1
        next_month_dates = df.index[(df.index.year == next_year) & (df.index.month == next_month)]
        if len(next_month_dates) == 0:
            return False
        first4 = next_month_dates[:4]

        return date == last_trading_day or date in first4

    def score(self, df: pd.DataFrame) -> pd.Series:
        """+1.5 in turn-of-month window, 0 outside. These 5 days historically capture most monthly gains."""
        scores = pd.Series(0.0, index=df.index)
        for i in range(len(df)):
            if self.is_in_window(df.index[i], df):
                scores.iloc[i] = 1.5
        return scores

    def signal(self, df: pd.DataFrame) -> pd.Series:
        """+1 in turn-of-month window, 0 otherwise."""
        signal = pd.Series(0, index=df.index)
        for i in range(len(df)):
            if self.is_in_window(df.index[i], df):
                signal.iloc[i] = 1
        return signal

    @staticmethod
    def description() -> str:
        return (
            "Turn of the Month Effect: Last trading day of month + first 4 trading days "
            "of next month. Disproportionate share of gains occur here. "
            "Driven by 401(k) inflows, month-end rebalancing, and institutional flows."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SUPER 8 DAYS
# ═══════════════════════════════════════════════════════════════════════════════

class Super8Days:
    """
    Super 8 Days (Almanac concept)
    - First 2 trading days of the month
    - Last 3 trading days of the month
    - Mid-month trading days 9, 10, 11

    These 8 days historically captured more gains than all other trading days
    combined for the DJIA. Trading only these 8 days would have outperformed
    buy-and-hold with much lower drawdowns.
    """

    def is_super8(self, date: pd.Timestamp, df: pd.DataFrame) -> bool:
        """Check if date is one of the Super 8 days."""
        month = date.month
        year = date.year
        month_dates = df.index[(df.index.year == year) & (df.index.month == month)]
        if len(month_dates) == 0:
            return False

        # Find the position of this date within the month
        positions = pd.Series(range(1, len(month_dates) + 1), index=month_dates)
        if date not in positions.index:
            return False
        pos = positions.loc[date]

        # First 2 trading days
        if pos in [1, 2]:
            return True

        # Last 3 trading days
        if pos > len(month_dates) - 3:
            return True

        # Mid-month days 9, 10, 11
        if pos in [9, 10, 11]:
            return True

        return False

    def score(self, df: pd.DataFrame) -> pd.Series:
        """+2 on Super 8 days, 0 on other days."""
        scores = pd.Series(0.0, index=df.index)
        for i in range(len(df)):
            if self.is_super8(df.index[i], df):
                scores.iloc[i] = 2.0
        return scores

    def signal(self, df: pd.DataFrame) -> pd.Series:
        """+1 on Super 8 days, 0 otherwise."""
        signal = pd.Series(0, index=df.index)
        for i in range(len(df)):
            if self.is_super8(df.index[i], df):
                signal.iloc[i] = 1
        return signal

    @staticmethod
    def description() -> str:
        return (
            "Super 8 Days: First 2 + last 3 + mid-month days 9-11. "
            "These 8 days captured more DJIA gains than all other days combined. "
            "Institutional flows and options expiration effects concentrate returns."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FIRST TRADING DAY OF THE MONTH PHENOMENON
# ═══════════════════════════════════════════════════════════════════════════════

class FirstTradingDay:
    """
    First Trading Day of the Month Phenomenon
    DJIA up 60% of first trading days, avg +0.25% since 1950.
    S&P 500 similar stats.

    Tied to 401(k) contributions, automatic investment plans, and
    institutional inflows hitting on the first of the month.
    """

    def is_first_trading_day(self, date: pd.Timestamp, df: pd.DataFrame) -> bool:
        month = date.month
        year = date.year
        month_dates = df.index[(df.index.year == year) & (df.index.month == month)]
        if len(month_dates) == 0:
            return False
        return date == month_dates[0]

    def score(self, df: pd.DataFrame) -> pd.Series:
        """+1 on first trading day, 0 otherwise. DJIA avg +0.25% first day since 1950."""
        scores = pd.Series(0.0, index=df.index)
        for i in range(len(df)):
            if self.is_first_trading_day(df.index[i], df):
                scores.iloc[i] = 1.0
        return scores

    def signal(self, df: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0, index=df.index)
        for i in range(len(df)):
            if self.is_first_trading_day(df.index[i], df):
                signal.iloc[i] = 1
        return signal

    @staticmethod
    def description() -> str:
        return (
            "First Trading Day of Month: DJIA positive 60% of time, "
            "avg +0.25% since 1950. Driven by 401(k) and automatic investment inflows."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-HOLIDAY EFFECT
# ═══════════════════════════════════════════════════════════════════════════════

class PreHolidayEffect:
    """
    Pre-Holiday Effect
    Trading day before a market holiday consistently shows higher-than-average returns.
    Avg return often 10x a normal trading day.
    Tied to short-covering, bullish sentiment ahead of holidays, and reduced volume.
    """

    def score(self, df: pd.DataFrame) -> pd.Series:
        """+1.5 on pre-holiday days, 0 otherwise."""
        scores = pd.Series(0.0, index=df.index)
        for i in range(len(df)):
            if _is_pre_holiday(df.index[i]):
                scores.iloc[i] = 1.5
        return scores

    def signal(self, df: pd.DataFrame) -> pd.Series:
        """+1 on pre-holiday days, 0 otherwise."""
        signal = pd.Series(0, index=df.index)
        for i in range(len(df)):
            if _is_pre_holiday(df.index[i]):
                signal.iloc[i] = 1
        return signal

    @staticmethod
    def description() -> str:
        return (
            "Pre-Holiday Effect: Day before market holidays avg return often 10x "
            "normal. Driven by short-covering, reduced volume, and positive sentiment."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PRESIDENTIAL ELECTION CYCLE (4-YEAR)
# ═══════════════════════════════════════════════════════════════════════════════

class PresidentialElectionCycle:
    """
    Presidential Election Cycle Theory (Yale Hirsch, 1967 Stock Trader's Almanac)

    Four-year cycle:
    - Year 1 (Post-Election): DJIA avg +3%, S&P avg +7.9%, weak
    - Year 2 (Midterm): DJIA avg +4%, S&P avg +4.6%, WEAK Apr-Sep especially
    - Year 3 (Pre-Election): DJIA avg +10.2%, S&P avg +17.2%, STRONGEST (up 90% of years)
    - Year 4 (Election): DJIA avg +6%, S&P avg +7.3%, stronger when incumbent runs

    The weak spot: Q2+Q3 of midterm years (Apr-Sep), historically the worst period.
    """

    def get_cycle_year(self, year: int) -> int:
        """
        Return cycle year (1-4) based on US presidential elections.
        Modern elections: 2020, 2024, 2028, 2032... (year % 4 == 0)
        Year 1 = post-election, Year 2 = midterm, Year 3 = pre-election, Year 4 = election.
        """
        rem = year % 4
        if rem == 0:
            return 4  # Election year
        elif rem == 1:
            return 1  # Post-election
        elif rem == 2:
            return 2  # Midterm
        else:  # rem == 3
            return 3  # Pre-election

    def get_cycle_score(self, cycle_year: int, month: int = None) -> float:
        """
        Base score for cycle year. Adjusted by month for midterm weakness.

        Scores:
        - Year 3 (Pre-Election): +2
        - Year 4 (Election): +1
        - Year 1 (Post-Election): 0
        - Year 2 (Midterm): -1 (worse in Apr-Sep: -2)
        """
        base_scores = {1: 0.0, 2: -1.0, 3: 2.0, 4: 1.0}
        score = base_scores.get(cycle_year, 0.0)

        # Midterm year Apr-Sep weakness: subtract extra 1
        if cycle_year == 2 and month and month in [4, 5, 6, 7, 8, 9]:
            score -= 1.0

        return score

    def score(self, df: pd.DataFrame) -> pd.Series:
        """Score based on presidential election cycle year and seasonal weakness."""
        scores = pd.Series(0.0, index=df.index)
        for i in range(len(df)):
            date = df.index[i]
            cycle_year = self.get_cycle_year(date.year)
            scores.iloc[i] = self.get_cycle_score(cycle_year, date.month)
        return scores

    def signal(self, df: pd.DataFrame) -> pd.Series:
        """+1 in pre-election/election years, -1 in midterm (esp Apr-Sep), 0 post-election."""
        scores = self.score(df)
        signal = pd.Series(0, index=df.index)
        signal[scores >= 1.5] = 1
        signal[scores <= -1.5] = -1
        return signal

    @staticmethod
    def description() -> str:
        return (
            "Presidential Election Cycle (Yale Hirsch 1967): "
            "Pre-Election Year (Year 3) strongest — DJIA avg +10.2%, S&P +17.2% (up 90% of years). "
            "Midterm Year (Year 2) weakest, especially Apr-Sep. "
            "Election Year (Year 4) avg +6-7%, better when incumbent runs."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SEPTEMBER / OCTOBER EFFECT
# ═══════════════════════════════════════════════════════════════════════════════

class SeptemberOctoberEffect:
    """
    September Effect: Historically worst month. DJIA avg -0.6%, S&P avg -0.7%.
    Only 42-45% positive months since 1950. Tied to mutual fund fiscal year-end
    (Oct 31), window dressing, and return to work after summer.

    October Effect: Volatile, crash-prone month (1929, 1987, 2008) but also
    historically marks many bear market bottoms. Known as the "Bear Killer."
    DJIA avg +0.5%, S&P avg +0.8% — not weak on average but with fat negative tails.
    """

    def score(self, df: pd.DataFrame) -> pd.Series:
        """September = -2 (strong caution), October = -0.5 (volatility premium but risky)."""
        scores = pd.Series(0.0, index=df.index)
        for i in range(len(df)):
            month = df.index[i].month
            if month == 9:
                scores.iloc[i] = -2.0
            elif month == 10:
                scores.iloc[i] = -0.5  # Slightly cautionary, not outright bearish
        return scores

    def signal(self, df: pd.DataFrame) -> pd.Series:
        """-1 in September, 0 in October, 0 otherwise."""
        signal = pd.Series(0, index=df.index)
        for i in range(len(df)):
            if df.index[i].month == 9:
                signal.iloc[i] = -1
        return signal

    @staticmethod
    def description() -> str:
        return (
            "September Effect: Worst month historically (DJIA -0.6%, S&P -0.7%, "
            "positive only 42-45% of time). October Effect: crash-prone but "
            "also 'Bear Killer' — many bear markets end in October. "
            "Volatile with fat negative tails despite positive average."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MID-MONTH BULGE
# ═══════════════════════════════════════════════════════════════════════════════

class MidMonthBulge:
    """
    Mid-Month Bulge: Trading days ~9-15 of each month tend to be stronger.
    After the first few days of the month settle, mid-month flows from
    automatic investment plans and options expiration positioning push prices.
    """

    def score(self, df: pd.DataFrame) -> pd.Series:
        """+1 on trading days 9-15, 0 otherwise."""
        scores = pd.Series(0.0, index=df.index)
        idx = df.index
        for year in pd.unique(idx.year):
            for mo in range(1, 13):
                yr_mo_mask = (idx.year == year) & (idx.month == mo)
                month_dates = idx[yr_mo_mask]
                if len(month_dates) < 15:
                    continue
                # Days 9-15 (positions 8-14 in 0-indexed)
                for date in month_dates[8:15]:
                    scores.loc[date] = 1.0
        return scores

    def signal(self, df: pd.DataFrame) -> pd.Series:
        """+1 on mid-month bulge days."""
        scores = self.score(df)
        signal = pd.Series(0, index=df.index)
        signal[scores >= 0.5] = 1
        return signal

    @staticmethod
    def description() -> str:
        return (
            "Mid-Month Bulge: Trading days 9-15 tend stronger due to "
            "mid-month institutional flows and options expiration positioning."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DECEMBER TAX-LOSS SELLING + JANUARY EFFECT
# ═══════════════════════════════════════════════════════════════════════════════

class TaxLossJanuaryEffect:
    """
    Tax-Loss Selling (December): Losers sold for tax purposes in December,
    creating downward pressure on beaten-down names, often followed by
    a January bounce as selling pressure abates.

    January Effect: Small caps historically outperform large caps in January,
    partly driven by tax-loss reversal. Weaker in recent decades but still
    observable in micro-cap names.
    """

    def score(self, df: pd.DataFrame) -> pd.Series:
        """
        +1 in January (January Effect bounce), -0.5 in December (tax-loss selling pressure).
        """
        scores = pd.Series(0.0, index=df.index)
        for i in range(len(df)):
            month = df.index[i].month
            if month == 1:
                scores.iloc[i] = 1.0
            elif month == 12:
                scores.iloc[i] = -0.5
        return scores

    def signal(self, df: pd.DataFrame) -> pd.Series:
        """+1 in January, 0 otherwise (December weakness is mild, not a short signal)."""
        signal = pd.Series(0, index=df.index)
        for i in range(len(df)):
            if df.index[i].month == 1:
                signal.iloc[i] = 1
        return signal

    @staticmethod
    def description() -> str:
        return (
            "Tax-Loss Selling + January Effect: December weakness from tax-loss "
            "harvesting, followed by January bounce (especially small caps). "
            "January Effect has weakened in recent decades but persists in micro-caps."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# COMPOSITE ALMANAC SCORER (ensembles all rules)
# ═══════════════════════════════════════════════════════════════════════════════

class AlmanacCompositeScorer:
    """
    Composite seasonal score combining all Stock Trader's Almanac rule sets.

    Weights calibrated to Almanac emphasis (Best Six Months is the flagship,
    followed by Election Cycle, monthly seasonality, and shorter-term patterns).

    Output: continuous score -5 to +5, suitable for:
    - Regime filtering (remove trades in low-score periods)
    - Position sizing (scale up in high-score, down in low-score)
    - Meta-labeling on top of other strategies
    """

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            "best_six_months_macd": 1.5,      # Flagship strategy
            "monthly_seasonality": 1.0,         # Monthly ranking
            "presidential_cycle": 1.2,          # 4-year cycle
            "september_october": 0.8,           # Worst months effect
            "turn_of_month": 0.5,               # Monthly bulge
            "super_8": 0.6,                     # Super 8 days
            "first_trading_day": 0.2,           # First day phenomenon
            "pre_holiday": 0.3,                 # Pre-holiday effect
            "santa_claus_rally": 0.4,           # SCR window
            "mid_month_bulge": 0.4,             # Mid-month strength
            "tax_loss_january": 0.3,            # Tax-loss / January
        }
        # Initialize sub-scorers
        self.monthly_seasonality = MonthlySeasonalitySignal("DJIA")
        self.best_six = BestSixMonthsMACDSwitch(use_nasdaq=False)
        self.presidential = PresidentialElectionCycle()
        self.sep_oct = SeptemberOctoberEffect()
        self.turn_of_month = TurnOfTheMonth()
        self.super_8 = Super8Days()
        self.first_td = FirstTradingDay()
        self.pre_holiday = PreHolidayEffect()
        self.scr = SantaClausRally()
        self.mid_month = MidMonthBulge()
        self.tax_loss = TaxLossJanuaryEffect()

    def compute_all_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all sub-scores and return as DataFrame."""
        df = _normalize_df(df)
        scores = pd.DataFrame(index=df.index)

        # Monthly seasonality (point-in-time)
        scores["monthly_seasonality"] = [self.monthly_seasonality.score(d) for d in df.index]

        # Best Six Months MACD
        scores["best_six_months_macd"] = self.best_six.score(df)

        # Presidential election cycle
        scores["presidential_cycle"] = self.presidential.score(df)

        # September/October
        scores["september_october"] = self.sep_oct.score(df)

        # Turn of month
        scores["turn_of_month"] = self.turn_of_month.score(df)

        # Super 8
        scores["super_8"] = self.super_8.score(df)

        # First trading day
        scores["first_trading_day"] = self.first_td.score(df)

        # Pre-holiday
        scores["pre_holiday"] = self.pre_holiday.score(df)

        # Santa Claus Rally
        scores["santa_claus_rally"] = self.scr.score(df)

        # Mid-month bulge
        scores["mid_month_bulge"] = self.mid_month.score(df)

        # Tax loss / January
        scores["tax_loss_january"] = self.tax_loss.score(df)

        return scores

    def composite_score(self, df: pd.DataFrame) -> pd.Series:
        """Compute weighted composite score."""
        all_scores = self.compute_all_scores(df)
        composite = pd.Series(0.0, index=df.index)
        for name, weight in self.weights.items():
            if name in all_scores.columns:
                composite += weight * all_scores[name]
        return composite

    def signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Composite binary signal:
        +1 (bullish seasonal) if composite > 1.0
        -1 (bearish seasonal) if composite < -1.0
        0 (neutral) otherwise
        """
        comp = self.composite_score(df)
        signal = pd.Series(0, index=df.index)
        signal[comp > 1.0] = 1
        signal[comp < -1.0] = -1
        return signal

    def regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Regime labels based on composite score:
        'strong_bull' (> 2.5), 'bull' (0.5 to 2.5), 'neutral' (-0.5 to 0.5),
        'bear' (-2.5 to -0.5), 'strong_bear' (< -2.5)
        """
        comp = self.composite_score(df)
        regime = pd.Series("neutral", index=df.index)
        regime[comp > 2.5] = "strong_bull"
        regime[(comp > 0.5) & (comp <= 2.5)] = "bull"
        regime[(comp >= -0.5) & (comp <= 0.5)] = "neutral"
        regime[(comp >= -2.5) & (comp < -0.5)] = "bear"
        regime[comp < -2.5] = "strong_bear"
        return regime

    @staticmethod
    def description() -> str:
        return (
            "Almanac Composite Scorer: Weighted ensemble of 11 seasonal/calendar rule sets "
            "from the Stock Trader's Almanac. Weights calibrated to flagship strategies "
            "(Best Six Months, Election Cycle, monthly seasonality dominant; "
            "shorter-term patterns supplementing). Score range typically -5 to +5."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# NASDAQ-SPECIFIC: NASDAQ BEST EIGHT MONTHS
# ═══════════════════════════════════════════════════════════════════════════════

class NasdaqBestEightMonths:
    """
    NASDAQ Best Eight Months (Nov-Jun)
    MACD-timed seasonal switch specific to NASDAQ. Slightly longer favorable
    season than DJIA/SPY due to tech seasonality.
    """

    def __init__(self):
        self.best_months = [11, 12, 1, 2, 3, 4, 5, 6]
        self.worst_months = [7, 8, 9, 10]

    def score(self, df: pd.DataFrame) -> pd.Series:
        df = _normalize_df(df)
        close = df["Close"]
        m = macd(close)
        scores = pd.Series(0.0, index=df.index)
        for i in range(len(df)):
            month = df.index[i].month
            if month in self.best_months:
                base = 1.0
                if m["macd"].iloc[i] > m["signal"].iloc[i]:
                    base = 2.0
                scores.iloc[i] = base
            elif month in self.worst_months:
                base = -1.0
                if m["macd"].iloc[i] < m["signal"].iloc[i]:
                    base = -2.0
                scores.iloc[i] = base
        return scores

    def signal(self, df: pd.DataFrame) -> pd.Series:
        scores = self.score(df)
        signal = pd.Series(0, index=df.index)
        signal[scores >= 1.0] = 1
        signal[scores <= -1.5] = -1
        return signal

    @staticmethod
    def description() -> str:
        return (
            "NASDAQ Best Eight Months (Nov-Jun): Tech sector's longer favorable season. "
            "MACD (12,26,9) times entry/exit. Almanac variant of Best Six Months."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# RULE SET EXPORT (for strategy system integration)
# ═══════════════════════════════════════════════════════════════════════════════

ALMANAC_RULE_SETS = {
    "s700_almanac_best_six_months": {
        "class": BestSixMonthsMACDSwitch,
        "description": BestSixMonthsMACDSwitch.description(),
        "category": "seasonal",
        "source": "Stock Trader's Almanac — Best Six Months Switching Strategy",
        "params": {"use_nasdaq": False},
    },
    "s701_almanac_nasdaq_eight_months": {
        "class": NasdaqBestEightMonths,
        "description": NasdaqBestEightMonths.description(),
        "category": "seasonal",
        "source": "Stock Trader's Almanac — NASDAQ Best Eight Months",
        "params": {},
    },
    "s702_almanac_trifecta": {
        "class": JanuaryBarometerTrifecta,
        "description": JanuaryBarometerTrifecta.description(),
        "category": "calendar",
        "source": "Stock Trader's Almanac — January Indicator Trifecta",
        "params": {},
    },
    "s703_almanac_election_cycle": {
        "class": PresidentialElectionCycle,
        "description": PresidentialElectionCycle.description(),
        "category": "calendar",
        "source": "Stock Trader's Almanac — Presidential Election Cycle",
        "params": {},
    },
    "s704_almanac_super8": {
        "class": Super8Days,
        "description": Super8Days.description(),
        "category": "calendar",
        "source": "Stock Trader's Almanac — Super 8 Days",
        "params": {},
    },
    "s705_almanac_turn_of_month": {
        "class": TurnOfTheMonth,
        "description": TurnOfTheMonth.description(),
        "category": "calendar",
        "source": "Stock Trader's Almanac — Turn of the Month Effect",
        "params": {},
    },
    "s706_almanac_composite": {
        "class": AlmanacCompositeScorer,
        "description": AlmanacCompositeScorer.description(),
        "category": "seasonal_ensemble",
        "source": "Stock Trader's Almanac — Composite All Rules",
        "params": {},
    },
    "s707_almanac_september_october": {
        "class": SeptemberOctoberEffect,
        "description": SeptemberOctoberEffect.description(),
        "category": "calendar",
        "source": "Stock Trader's Almanac — September/October Effects",
        "params": {},
    },
    "s708_almanac_santa_claus": {
        "class": SantaClausRally,
        "description": SantaClausRally.description(),
        "category": "calendar",
        "source": "Stock Trader's Almanac — Santa Claus Rally",
        "params": {"threshold_pct": 0.5},
    },
    "s709_almanac_pre_holiday": {
        "class": PreHolidayEffect,
        "description": PreHolidayEffect.description(),
        "category": "calendar",
        "source": "Stock Trader's Almanac — Pre-Holiday Effect",
        "params": {},
    },
    "s710_almanac_first_trading_day": {
        "class": FirstTradingDay,
        "description": FirstTradingDay.description(),
        "category": "calendar",
        "source": "Stock Trader's Almanac — First Trading Day Phenomenon",
        "params": {},
    },
    "s711_almanac_monthly_seasonality": {
        "class": MonthlySeasonalitySignal,
        "description": MonthlySeasonalitySignal.description(),
        "category": "seasonal",
        "source": "Stock Trader's Almanac — Monthly Seasonality Rankings",
        "params": {"index": "DJIA"},
    },
    "s712_almanac_mid_month_bulge": {
        "class": MidMonthBulge,
        "description": MidMonthBulge.description(),
        "category": "calendar",
        "source": "Stock Trader's Almanac — Mid-Month Bulge",
        "params": {},
    },
    "s713_almanac_tax_loss_january": {
        "class": TaxLossJanuaryEffect,
        "description": TaxLossJanuaryEffect.description(),
        "category": "seasonal",
        "source": "Stock Trader's Almanac — Tax-Loss Selling / January Effect",
        "params": {},
    },
}


def generate_all_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate all Almanac signals for the given price data.
    Returns DataFrame with all signals as columns.
    """
    df = _normalize_df(df)
    signals = pd.DataFrame(index=df.index)

    for strategy_id, info in ALMANAC_RULE_SETS.items():
        try:
            cls = info["class"]
            params = info.get("params", {})
            instance = cls(**params)
            signal = instance.signal(df)
            signals[strategy_id] = signal
        except Exception as e:
            print(f"Warning: {strategy_id} failed: {e}")
            signals[strategy_id] = 0

    return signals


def almanac_dashboard(df: pd.DataFrame, year: int = None) -> str:
    """
    Generate a human-readable Almanac dashboard for a given year or the latest data.
    Returns a formatted string with seasonal conditions and signals.
    """
    df = _normalize_df(df)
    if year is None:
        year = df.index[-1].year

    composite = AlmanacCompositeScorer()
    scores = composite.compute_all_scores(df)
    regime = composite.regime(df)

    # Filter to requested year
    yr_mask = df.index.year == year

    lines = []
    lines.append(f"╔══════════════════════════════════════════════════════╗")
    lines.append(f"║  STOCK TRADER'S ALMANAC — {year} SEASONAL DASHBOARD  ║")
    lines.append(f"╚══════════════════════════════════════════════════════╝")
    lines.append("")

    # Election cycle
    cycle = PresidentialElectionCycle()
    cycle_year = cycle.get_cycle_year(year)
    cycle_names = {1: "Post-Election", 2: "Midterm", 3: "Pre-Election", 4: "Election"}
    lines.append(f"Presidential Cycle: Year {cycle_year} — {cycle_names.get(cycle_year, 'Unknown')}")
    lines.append(f"  → Cycle Score: {cycle.get_cycle_score(cycle_year):+.1f}")

    # Monthly breakdown
    lines.append("")
    lines.append(f"{'Month':<12} {'Seasonal':>8} {'Cycle':>8} {'Best6M':>8} {'Signal':>8}")
    lines.append("-" * 50)

    monthly = MonthlySeasonalitySignal("DJIA")
    best6 = BestSixMonthsMACDSwitch()

    for m in range(1, 13):
        month_name = monthly.seasonality[m]["name"]
        month_data = scores[yr_mask & (df.index.month == m)]
        if len(month_data) > 0:
            seas_score = month_data["monthly_seasonality"].mean()
            cycle_score = month_data["presidential_cycle"].mean()
            best6_score = month_data["best_six_months_macd"].mean()
            avg_sig = composite.signal(df.loc[yr_mask & (df.index.month == m)]).mean()
            lines.append(f"{month_name:<12} {seas_score:>+8.2f} {cycle_score:>+8.2f} {best6_score:>+8.2f} {avg_sig:>+8.2f}")

    # Summary
    lines.append("")
    yr_regime = regime[yr_mask].value_counts().to_dict()
    lines.append(f"Year Regime Distribution: {yr_regime}")
    lines.append(f"Average Composite Score: {composite.composite_score(df.loc[yr_mask]).mean():+.2f}")

    # Key dates
    lines.append("")
    lines.append("Key Windows:")
    scr = SantaClausRally()
    scr_dates = [d for d in df.index if scr.is_in_window(d) and d.year == year]
    lines.append(f"  Santa Claus Rally: {len(scr_dates)} trading days in window")

    totm = TurnOfTheMonth()
    totm_dates = [d for d in df.index if totm.is_in_window(d, df) and d.year == year]
    lines.append(f"  Turn of Month days: {len(totm_dates)}")

    return "\n".join(lines)


if __name__ == "__main__":
    print("Stock Trader's Almanac Rule Sets loaded.")
    print(f"Available strategies: {len(ALMANAC_RULE_SETS)}")
    for sid in sorted(ALMANAC_RULE_SETS.keys()):
        print(f"  {sid}: {ALMANAC_RULE_SETS[sid]['category']}")
