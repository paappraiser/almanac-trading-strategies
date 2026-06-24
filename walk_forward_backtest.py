#!/usr/bin/env python3
"""
Stock Trader's Almanac — Walk-Forward Backtest with Optimization
=================================================================
Rolling walk-forward validation with per-window weight optimization.
Tests whether Almanac seasonal signals survive out-of-sample and
whether optimized weights beat the default composite.

Methodology:
- 5-year training window, 1-year OOS step
- Optimize composite weights on training window (maximize Sharpe)
- Apply optimized weights to OOS window
- Compare: Optimized vs Default (equal-weighted) vs Buy & Hold
- Test on: SPY, QQQ, DIA

Data: 2000-01-01 through today (last pull date in header)
"""
import yfinance as yf
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import Almanac rules
import sys
sys.path.insert(0, '/root/.hermes/aqrs')
from stock_traders_almanac_rules import (
    _normalize_df, AlmanacCompositeScorer,
    MonthlySeasonalitySignal, BestSixMonthsMACDSwitch, PresidentialElectionCycle,
    SeptemberOctoberEffect, TurnOfTheMonth, Super8Days, FirstTradingDay,
    PreHolidayEffect, SantaClausRally, MidMonthBulge, TaxLossJanuaryEffect,
    NasdaqBestEightMonths,
)

TICKERS = ["SPY", "QQQ", "DIA"]
TRAIN_YEARS = 5
OOS_YEARS = 1
RISK_FREE = 0.04
COST_BPS = 5  # 5 bps per trade for signal flips

# Signal component names (match AlmanacCompositeScorer weights)
SIGNAL_NAMES = [
    "best_six_months_macd",
    "monthly_seasonality",
    "presidential_cycle",
    "september_october",
    "turn_of_month",
    "super_8",
    "first_trading_day",
    "pre_holiday",
    "santa_claus_rally",
    "mid_month_bulge",
    "tax_loss_january",
]


def compute_sub_signals(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Compute all 11 sub-signal scores for the full dataset."""
    scores = pd.DataFrame(index=df.index)

    ms = MonthlySeasonalitySignal("DJIA")
    b6 = BestSixMonthsMACDSwitch(use_nasdaq=(ticker == "QQQ"))
    pc = PresidentialElectionCycle()
    so = SeptemberOctoberEffect()
    tom = TurnOfTheMonth()
    s8 = Super8Days()
    ftd = FirstTradingDay()
    ph = PreHolidayEffect()
    scr = SantaClausRally()
    mm = MidMonthBulge()
    tl = TaxLossJanuaryEffect()

    scores["best_six_months_macd"] = b6.score(df)
    scores["monthly_seasonality"] = pd.Series([ms.score(d) for d in df.index], index=df.index)
    scores["presidential_cycle"] = pc.score(df)
    scores["september_october"] = so.score(df)
    scores["turn_of_month"] = tom.score(df)
    scores["super_8"] = s8.score(df)
    scores["first_trading_day"] = ftd.score(df)
    scores["pre_holiday"] = ph.score(df)
    scores["santa_claus_rally"] = scr.score(df)
    scores["mid_month_bulge"] = mm.score(df)
    scores["tax_loss_january"] = tl.score(df)

    return scores


def annualized_sharpe(returns: pd.Series) -> float:
    """Annualized Sharpe ratio."""
    ann_ret = (1 + returns.mean()) ** 252 - 1
    ann_vol = returns.std() * np.sqrt(252)
    if ann_vol < 0.001:
        return 0.0
    return (ann_ret - RISK_FREE) / ann_vol


def annualized_return(returns: pd.Series) -> float:
    return (1 + returns.mean()) ** 252 - 1


def max_drawdown(returns: pd.Series) -> float:
    eq = (1 + returns).cumprod()
    return float((eq / eq.cummax() - 1).min())


def calmar_ratio(returns: pd.Series) -> float:
    dd = max_drawdown(returns)
    if abs(dd) < 0.001:
        return annualized_return(returns) / 0.001
    return annualized_return(returns) / abs(dd)


def signal_to_positions(signal: pd.Series, threshold: float = 1.0) -> pd.Series:
    """Convert composite score to positions: +1 (long), 0 (flat), -1 (short)."""
    pos = pd.Series(0, index=signal.index)
    pos[signal > threshold] = 1
    pos[signal < -threshold] = -1
    return pos


def strategy_returns(df: pd.DataFrame, sub_signals: pd.DataFrame, weights: dict,
                     threshold: float = 1.0) -> dict:
    """Compute strategy returns given weights and threshold."""
    rets = df["Close"].pct_change()

    # Build composite score
    composite = pd.Series(0.0, index=df.index)
    for name, w in weights.items():
        if name in sub_signals.columns:
            composite += w * sub_signals[name]

    # Convert to positions
    positions = signal_to_positions(composite, threshold)
    pos_shifted = positions.shift(1).fillna(0)

    # Strategy returns (long only — if position=1, capture return; position=-1 means flat/cash)
    strat_rets = rets * pos_shifted.clip(lower=0)

    # Transaction costs: cost when position changes
    pos_change = pos_shifted.diff().abs()
    cost_series = pos_change * (COST_BPS / 10000)
    strat_rets_net = strat_rets - cost_series

    # Metrics
    n_long = int((positions == 1).sum())
    n_total = len(df)
    exposure = n_long / max(n_total, 1)

    total_ret = float((1 + strat_rets_net).prod() - 1)
    ann_ret = annualized_return(strat_rets_net)
    sharpe = annualized_sharpe(strat_rets_net)
    dd = max_drawdown(strat_rets_net)
    calmar = calmar_ratio(strat_rets_net)
    vol = float(strat_rets_net.std() * np.sqrt(252))

    # Trade count
    trades = int((pos_change > 0).sum())

    return {
        "total_return": total_ret,
        "ann_return": ann_ret,
        "sharpe": sharpe,
        "max_drawdown": dd,
        "calmar": calmar,
        "ann_vol": vol,
        "exposure": exposure,
        "trades": trades,
        "n_long": n_long,
        "n_total": n_total,
    }


def optimize_weights(sub_signals: pd.DataFrame, rets: pd.Series,
                     threshold: float = 1.0) -> dict:
    """
    Optimize composite weights on training data.
    Grid search over weight ranges for each signal.
    Returns best weights dict and best Sharpe.
    """
    # Search space: weight ranges for each signal
    weight_grid = {
        "best_six_months_macd": [0.5, 1.0, 1.5, 2.0],
        "monthly_seasonality": [0.5, 1.0, 1.5],
        "presidential_cycle": [0.5, 1.0, 1.5, 2.0],
        "september_october": [0.3, 0.5, 0.8, 1.0],
        "turn_of_month": [0.0, 0.3, 0.5, 0.8],
        "super_8": [0.0, 0.3, 0.6, 1.0],
        "first_trading_day": [0.0, 0.1, 0.2, 0.3],
        "pre_holiday": [0.0, 0.2, 0.3, 0.5],
        "santa_claus_rally": [0.0, 0.2, 0.4, 0.6],
        "mid_month_bulge": [0.0, 0.2, 0.4, 0.6],
        "tax_loss_january": [0.0, 0.1, 0.2, 0.3],
    }

    # Build all combinations (pruned: only vary top 6 signals, freeze others)
    # Full grid would be 4^11 = 4M combos. We vary the top signals.
    freeze_weights = {
        "first_trading_day": 0.2,
        "pre_holiday": 0.3,
        "santa_claus_rally": 0.4,
        "mid_month_bulge": 0.4,
        "tax_loss_january": 0.3,
    }

    best_sharpe = -999
    best_weights = None
    best_threshold = threshold

    # Grid over main signals
    main_signals = ["best_six_months_macd", "monthly_seasonality", "presidential_cycle",
                    "september_october", "turn_of_month", "super_8"]

    # Pruned grid: ~4^6 * 3^2 = ~37k combos, manageable
    grid_iter = 0
    for b6_w in weight_grid["best_six_months_macd"]:
        for ms_w in weight_grid["monthly_seasonality"]:
            for pc_w in weight_grid["presidential_cycle"]:
                for so_w in weight_grid["september_october"]:
                    for tom_w in weight_grid["turn_of_month"]:
                        for s8_w in weight_grid["super_8"]:
                            # Also try threshold variations
                            for thresh in [0.5, 1.0, 1.5]:
                                test_weights = {
                                    "best_six_months_macd": b6_w,
                                    "monthly_seasonality": ms_w,
                                    "presidential_cycle": pc_w,
                                    "september_october": so_w,
                                    "turn_of_month": tom_w,
                                    "super_8": s8_w,
                                    **freeze_weights,
                                }
                                grid_iter += 1
                                if grid_iter % 5000 == 0:
                                    pass  # progress indicator (silent)

                                composite = pd.Series(0.0, index=sub_signals.index)
                                for name, w in test_weights.items():
                                    composite += w * sub_signals[name]

                                positions = signal_to_positions(composite, thresh)
                                pos_shifted = positions.shift(1).fillna(0)
                                strat_rets = rets * pos_shifted.clip(lower=0)
                                pos_change = pos_shifted.diff().abs()
                                cost = pos_change * (COST_BPS / 10000)
                                strat_rets_net = strat_rets - cost

                                if len(strat_rets_net) < 50:
                                    continue
                                sharpe = annualized_sharpe(strat_rets_net)
                                if sharpe > best_sharpe:
                                    best_sharpe = sharpe
                                    best_weights = test_weights.copy()
                                    best_threshold = thresh

    return best_weights, best_threshold, best_sharpe


def walk_forward(df: pd.DataFrame, ticker: str) -> dict:
    """Run full walk-forward backtest with optimization."""
    sub_signals = compute_sub_signals(df, ticker)
    rets = df["Close"].pct_change()

    # Default weights (from AlmanacCompositeScorer)
    default_weights = {
        "best_six_months_macd": 1.5,
        "monthly_seasonality": 1.0,
        "presidential_cycle": 1.2,
        "september_october": 0.8,
        "turn_of_month": 0.5,
        "super_8": 0.6,
        "first_trading_day": 0.2,
        "pre_holiday": 0.3,
        "santa_claus_rally": 0.4,
        "mid_month_bulge": 0.4,
        "tax_loss_january": 0.3,
    }

    years = sorted(set(df.index.year))
    first_year = years[0]
    last_year = years[-1]

    windows = []
    opt_weights_history = []
    opt_thresholds_history = []

    for oos_start_year in range(first_year + TRAIN_YEARS, last_year + 1, OOS_YEARS):
        train_start = oos_start_year - TRAIN_YEARS
        train_end = oos_start_year - 1
        oos_end = min(oos_start_year + OOS_YEARS - 1, last_year)

        train_mask = (df.index.year >= train_start) & (df.index.year <= train_end)
        oos_mask = (df.index.year >= oos_start_year) & (df.index.year <= oos_end)

        if train_mask.sum() < 200 or oos_mask.sum() < 50:
            continue

        train_sigs = sub_signals.loc[train_mask]
        train_rets = rets.loc[train_mask]
        oos_sigs = sub_signals.loc[oos_mask]
        oos_rets = rets.loc[oos_mask]

        # Optimize on training window
        opt_weights, opt_thresh, train_sharpe = optimize_weights(train_sigs, train_rets)
        if opt_weights is None:
            continue

        opt_weights_history.append({
            "train_window": f"{train_start}-{train_end}",
            "oos_window": f"{oos_start_year}-{oos_end}",
            "weights": opt_weights,
            "threshold": opt_thresh,
            "train_sharpe": train_sharpe,
        })

        # Test on OOS: optimized weights
        opt_result = strategy_returns(
            df.loc[oos_mask], oos_sigs, opt_weights, opt_thresh
        )
        opt_result["type"] = "optimized"
        opt_result["window"] = f"{oos_start_year}-{oos_end}"

        # Test on OOS: default weights
        def_result = strategy_returns(
            df.loc[oos_mask], oos_sigs, default_weights, 1.0
        )
        def_result["type"] = "default"
        def_result["window"] = f"{oos_start_year}-{oos_end}"

        # Buy & hold
        bh_rets = oos_rets
        bh_total = float((1 + bh_rets).prod() - 1)
        bh_result = {
            "type": "buy_hold",
            "window": f"{oos_start_year}-{oos_end}",
            "total_return": bh_total,
            "ann_return": annualized_return(bh_rets),
            "sharpe": annualized_sharpe(bh_rets),
            "max_drawdown": max_drawdown(bh_rets),
            "calmar": calmar_ratio(bh_rets),
            "ann_vol": float(bh_rets.std() * np.sqrt(252)),
            "exposure": 1.0,
            "trades": 0,
            "n_total": len(bh_rets),
            "n_long": len(bh_rets),
        }

        windows.append(opt_result)
        windows.append(def_result)
        windows.append(bh_result)

    # Full-period all-in backtest (no walk-forward, just optimized on everything)
    # This is informational only — in-sample
    full_opt_weights, full_opt_thresh, full_sharpe = optimize_weights(sub_signals, rets)
    full_opt_result = strategy_returns(df, sub_signals, full_opt_weights, full_opt_thresh)
    full_opt_result["type"] = "full_sample_optimized"
    full_opt_result["window"] = f"{first_year}-{last_year}"

    full_def_result = strategy_returns(df, sub_signals, default_weights, 1.0)
    full_def_result["type"] = "full_sample_default"
    full_def_result["window"] = f"{first_year}-{last_year}"

    bh_total = float((1 + rets).prod() - 1)
    full_bh_result = {
        "type": "full_sample_buy_hold",
        "window": f"{first_year}-{last_year}",
        "total_return": bh_total,
        "ann_return": annualized_return(rets),
        "sharpe": annualized_sharpe(rets),
        "max_drawdown": max_drawdown(rets),
        "calmar": calmar_ratio(rets),
        "ann_vol": float(rets.std() * np.sqrt(252)),
        "exposure": 1.0,
        "trades": 0,
        "n_total": len(rets),
        "n_long": len(rets),
    }

    # Aggregate OOS windows
    oos_opt = [w for w in windows if w["type"] == "optimized"]
    oos_def = [w for w in windows if w["type"] == "default"]
    oos_bh = [w for w in windows if w["type"] == "buy_hold"]

    def aggregate(windows_list):
        if not windows_list:
            return {}
        # Weight by n_total in each window
        total_n = sum(w["n_total"] for w in windows_list)
        if total_n == 0:
            return {}
        avg = {}
        for key in ["total_return", "ann_return", "sharpe", "max_drawdown", "calmar", "ann_vol", "exposure"]:
            vals = [w[key] * w["n_total"] / total_n for w in windows_list]
            avg[key] = sum(vals)
        avg["total_trades"] = sum(w.get("trades", 0) for w in windows_list)
        avg["n_total"] = total_n
        avg["n_windows"] = len(windows_list)
        return avg

    agg_opt = aggregate(oos_opt)
    agg_def = aggregate(oos_def)
    agg_bh = aggregate(oos_bh)

    # Final optimized weights (average across all windows, or full-sample)
    avg_weights = {}
    if opt_weights_history:
        for name in SIGNAL_NAMES:
            vals = [h["weights"].get(name, 0) for h in opt_weights_history]
            avg_weights[name] = np.mean(vals)
        avg_threshold = np.mean([h["threshold"] for h in opt_weights_history])
    else:
        avg_weights = default_weights
        avg_threshold = 1.0

    return {
        "ticker": ticker,
        "period": f"{first_year}-{last_year}",
        "n_windows": len(oos_opt),
        "full_sample": {
            "optimized": full_opt_result,
            "default": full_def_result,
            "buy_hold": full_bh_result,
        },
        "oos_aggregate": {
            "optimized": agg_opt,
            "default": agg_def,
            "buy_hold": agg_bh,
        },
        "oos_windows": windows,
        "weights_history": opt_weights_history,
        "optimized_weights": avg_weights,
        "optimized_threshold": avg_threshold,
        "default_weights": default_weights,
        "full_optimized_weights": full_opt_weights,
        "full_optimized_threshold": full_opt_thresh,
    }


def print_results(results: dict):
    """Pretty print the backtest results."""
    ticker = results["ticker"]
    period = results["period"]
    n_w = results["n_windows"]

    print(f"\n{'='*70}")
    print(f"  {ticker} — WALK-FORWARD BACKTEST ({period}, {n_w} OOS windows)")
    print(f"{'='*70}")

    # Full sample (in-sample reference)
    fs = results["full_sample"]
    print(f"\n  FULL SAMPLE (in-sample, informational only):")
    print(f"  {'Metric':<20} {'Optimized':>12} {'Default':>12} {'Buy&Hold':>12}")
    print(f"  {'-'*56}")
    for metric, label in [
        ("ann_return", "Ann Return"), ("sharpe", "Sharpe"),
        ("max_drawdown", "Max DD"), ("calmar", "Calmar"),
        ("ann_vol", "Ann Vol"), ("exposure", "Exposure"),
    ]:
        ov = fs["optimized"].get(metric, 0)
        dv = fs["default"].get(metric, 0)
        bv = fs["buy_hold"].get(metric, 0)
        if metric in ("ann_return", "ann_vol", "max_drawdown", "exposure"):
            print(f"  {label:<20} {ov*100:>+11.1f}% {dv*100:>+11.1f}% {bv*100:>+11.1f}%")
        else:
            print(f"  {label:<20} {ov:>12.2f} {dv:>12.2f} {bv:>12.2f}")

    # OOS aggregate
    oos = results["oos_aggregate"]
    print(f"\n  OUT-OF-SAMPLE AGGREGATE ({n_w} windows):")
    print(f"  {'Metric':<20} {'Optimized':>12} {'Default':>12} {'Buy&Hold':>12}")
    print(f"  {'-'*56}")
    for metric, label in [
        ("ann_return", "Ann Return"), ("sharpe", "Sharpe"),
        ("max_drawdown", "Max DD"), ("calmar", "Calmar"),
        ("ann_vol", "Ann Vol"), ("exposure", "Exposure"),
    ]:
        ov = oos["optimized"].get(metric, 0)
        dv = oos["default"].get(metric, 0)
        bv = oos["buy_hold"].get(metric, 0)
        if metric in ("ann_return", "ann_vol", "max_drawdown", "exposure"):
            print(f"  {label:<20} {ov*100:>+11.1f}% {dv*100:>+11.1f}% {bv*100:>+11.1f}%")
        else:
            print(f"  {label:<20} {ov:>12.2f} {dv:>12.2f} {bv:>12.2f}")

    # Per-window breakdown
    print(f"\n  PER-WINDOW OOS RETURNS:")
    print(f"  {'Window':<14} {'Optimized':>10} {'Default':>10} {'Buy&Hold':>10} {'Win?':>6}")
    print(f"  {'-'*54}")
    opt_wins = 0
    def_wins = 0
    for w_opt, w_def, w_bh in zip(
        [w for w in results["oos_windows"] if w["type"] == "optimized"],
        [w for w in results["oos_windows"] if w["type"] == "default"],
        [w for w in results["oos_windows"] if w["type"] == "buy_hold"],
    ):
        win = w_opt["window"]
        o = w_opt["ann_return"]
        d = w_def["ann_return"]
        b = w_bh["ann_return"]
        best = max(o, d, b)
        marker = "OPT" if o == best else ("DEF" if d == best else "B&H")
        if o > b:
            opt_wins += 1
        if d > b:
            def_wins += 1
        print(f"  {win:<14} {o*100:>+9.1f}% {d*100:>+9.1f}% {b*100:>+9.1f}% {marker:>6}")

    n = len([w for w in results["oos_windows"] if w["type"] == "optimized"])
    print(f"\n  Optimized beat B&H in {opt_wins}/{n} OOS windows")
    print(f"  Default beat B&H in {def_wins}/{n} OOS windows")

    # Optimized weights
    print(f"\n  OPTIMIZED WEIGHTS (average across all windows):")
    print(f"  Optimal threshold: {results['optimized_threshold']:.2f}")
    print(f"  {'Signal':<25} {'Default':>8} {'Optimized':>8} {'Delta':>8}")
    print(f"  {'-'*53}")
    for name in SIGNAL_NAMES:
        dw = results["default_weights"].get(name, 0)
        ow = results["optimized_weights"].get(name, 0)
        delta = ow - dw
        marker = " ▲" if delta > 0.1 else (" ▼" if delta < -0.1 else "  ")
        print(f"  {name:<25} {dw:>8.2f} {ow:>8.2f} {delta:>+8.2f}{marker}")

    # Full period optimized weights (info only)
    print(f"\n  FULL-SAMPLE OPTIMIZED WEIGHTS (in-sample, informational):")
    print(f"  Threshold: {results['full_optimized_threshold']:.2f}")
    for name in SIGNAL_NAMES:
        dw = results["default_weights"].get(name, 0)
        fw = results["full_optimized_weights"].get(name, 0)
        print(f"  {name:<25} {dw:>8.2f} → {fw:>8.2f}")


def main():
    now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    print(f"Stock Trader's Almanac — Walk-Forward Backtest")
    print(f"Run: {now}")
    print(f"Costs: {COST_BPS}bps per trade, {RISK_FREE*100:.0f}% risk-free")
    print(f"Train: {TRAIN_YEARS}yr, OOS step: {OOS_YEARS}yr")
    print(f"Tickers: {TICKERS}")

    all_results = {}
    for ticker in TICKERS:
        print(f"\n  Loading {ticker}...", end=" ", flush=True)
        df_raw = yf.download(ticker, start="2000-01-01", progress=False)
        df = _normalize_df(df_raw)
        print(f"{len(df)} rows, {df.index[0].date()} → {df.index[-1].date()}")

        results = walk_forward(df, ticker)
        all_results[ticker] = results
        print_results(results)

    # Cross-ticker summary
    print(f"\n{'='*70}")
    print(f"  CROSS-TICKER SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Ticker':<8} {'OOS Opt':>10} {'OOS Def':>10} {'OOS B&H':>10} {'Opt Sharpe':>10}")
    print(f"  {'-'*52}")
    for ticker in TICKERS:
        r = all_results[ticker]
        oo = r["oos_aggregate"]["optimized"].get("ann_return", 0)
        od = r["oos_aggregate"]["default"].get("ann_return", 0)
        ob = r["oos_aggregate"]["buy_hold"].get("ann_return", 0)
        os = r["oos_aggregate"]["optimized"].get("sharpe", 0)
        print(f"  {ticker:<8} {oo*100:>+9.1f}% {od*100:>+9.1f}% {ob*100:>+9.1f}% {os:>10.2f}")

    print(f"\n  {'='*70}")
    print(f"  CONCLUSION:")
    # Count wins
    total_wins_opt = sum(
        sum(1 for w in all_results[t]["oos_windows"]
            if w["type"] == "optimized" and w["ann_return"] > 0)
        for t in TICKERS
    )
    total_wins_bh = sum(
        sum(1 for w in all_results[t]["oos_windows"]
            if w["type"] == "buy_hold" and w["ann_return"] > 0)
        for t in TICKERS
    )
    print(f"  Optimized positive OOS windows: {total_wins_opt}")
    print(f"  Buy & Hold positive OOS windows: {total_wins_bh}")
    print(f"  Data through: {now}")

    return all_results


if __name__ == "__main__":
    all_results = main()
