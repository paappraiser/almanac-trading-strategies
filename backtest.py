#!/usr/bin/env python3
"""
Almanac Trading Strategies — Quick Backtest
============================================
Validates all 14 Almanac rule sets against real market data.
Outputs seasonal edge metrics vs buy-and-hold.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from stock_traders_almanac import (
    AlmanacCompositeScorer,
    BestSixMonthsMACDSwitch,
    NasdaqBestEightMonths,
    generate_all_signals,
    almanac_dashboard,
)

TICKERS = ["DIA", "SPY", "QQQ"]
START = "2005-01-01"
END = "2025-01-01"


def run_backtest(ticker: str) -> dict:
    """Run full Almanac backtest for a single ticker."""
    df_raw = yf.download(ticker, start=START, end=END, auto_adjust=False, progress=False)
    from stock_traders_almanac import _normalize_df
    df = _normalize_df(df_raw)
    rets = df["Close"].pct_change()

    # Composite signal
    composite = AlmanacCompositeScorer()
    sig = composite.signal(df)

    results = {"ticker": ticker, "rows": len(df)}

    # Regime returns
    for label, mask_val in [("bull", 1), ("bear", -1), ("neutral", 0)]:
        mask = sig == mask_val
        n = int(mask.sum())
        total_ret = float((1 + rets[mask]).prod() - 1)
        ann_ret = (1 + total_ret) ** (252 / max(n, 1)) - 1
        results[f"{label}_days"] = n
        results[f"{label}_total"] = total_ret
        results[f"{label}_ann"] = ann_ret

    # Buy & hold
    bh_total = float((1 + rets).prod() - 1)
    bh_ann = (1 + bh_total) ** (252 / len(df)) - 1
    results["bh_total"] = bh_total
    results["bh_ann"] = bh_ann

    # Best Six Months standalone
    best6 = BestSixMonthsMACDSwitch(use_nasdaq=(ticker == "QQQ"))
    bsig = best6.signal(df)
    for label, mask_val in [("b6m_long", 1), ("b6m_cash", -1)]:
        mask = bsig == mask_val
        n = int(mask.sum())
        total_ret = float((1 + rets[mask]).prod() - 1)
        ann_ret = (1 + total_ret) ** (252 / max(n, 1)) - 1
        results[f"{label}_days"] = n
        results[f"{label}_ann"] = ann_ret

    # Nasdaq Eight Months (QQQ only)
    if ticker == "QQQ":
        nasdaq = NasdaqBestEightMonths()
        nsig = nasdaq.signal(df)
        for label, mask_val in [("n8m_long", 1), ("n8m_cash", -1)]:
            mask = nsig == mask_val
            n = int(mask.sum())
            total_ret = float((1 + rets[mask]).prod() - 1)
            ann_ret = (1 + total_ret) ** (252 / max(n, 1)) - 1
            results[f"{label}_days"] = n
            results[f"{label}_ann"] = ann_ret

    return results


def main():
    print("=" * 70)
    print("  STOCK TRADER'S ALMANAC — FULL BACKTEST")
    print(f"  Period: {START} to {END}")
    print("=" * 70)

    all_results = []
    for ticker in TICKERS:
        print(f"\n  Loading {ticker}...", end=" ", flush=True)
        results = run_backtest(ticker)
        all_results.append(results)
        print(f"{results['rows']} trading days")

        print(f"\n  {'─' * 50}")
        print(f"  {ticker} — Composite Seasonal Signal")
        print(f"  {'─' * 50}")
        print(f"  Bull regime:  {results['bull_days']:>5d} days  "
              f"{results['bull_total']*100:>+7.1f}% total  {results['bull_ann']*100:>+6.1f}% ann")
        print(f"  Bear regime:  {results['bear_days']:>5d} days  "
              f"{results['bear_total']*100:>+7.1f}% total  {results['bear_ann']*100:>+6.1f}% ann")
        print(f"  Neutral:      {results['neutral_days']:>5d} days  "
              f"{results['neutral_total']*100:>+7.1f}% total  {results['neutral_ann']*100:>+6.1f}% ann")
        print(f"  Buy & Hold:   {results['rows']:>5d} days  "
              f"{results['bh_total']*100:>+7.1f}% total  {results['bh_ann']*100:>+6.1f}% ann")

        print(f"\n  {ticker} — Best Six Months MACD Switch")
        print(f"  Long (Nov-Apr): {results['b6m_long_days']:>5d} days  {results['b6m_long_ann']*100:>+6.1f}% ann")
        print(f"  Cash (May-Oct): {results['b6m_cash_days']:>5d} days  {results['b6m_cash_ann']*100:>+6.1f}% ann")

        if ticker == "QQQ":
            print(f"\n  {ticker} — NASDAQ Best Eight Months")
            print(f"  Long (Nov-Jun): {results['n8m_long_days']:>5d} days  {results['n8m_long_ann']*100:>+6.1f}% ann")
            print(f"  Cash (Jul-Oct): {results['n8m_cash_days']:>5d} days  {results['n8m_cash_ann']*100:>+6.1f}% ann")

    # Summary table
    print(f"\n  {'=' * 70}")
    print(f"  SUMMARY: Seasonal Edge vs Buy & Hold")
    print(f"  {'=' * 70}")
    print(f"  {'Ticker':<6} {'Bull Ann':>10} {'Bear Ann':>10} {'B&H Ann':>10} {'Edge':>10}")
    print(f"  {'─' * 50}")
    for r in all_results:
        edge = r["bull_ann"] - r["bh_ann"]
        print(f"  {r['ticker']:<6} {r['bull_ann']*100:>+9.1f}% {r['bear_ann']*100:>+9.1f}% "
              f"{r['bh_ann']*100:>+9.1f}% {edge*100:>+9.1f}%")

    # Dashboard for latest year
    print(f"\n  {'=' * 70}")
    print(f"  2024 SEASONAL DASHBOARD (SPY)")
    print(f"  {'=' * 70}")
    spy = yf.download("SPY", start="2024-01-01", end="2025-01-01", auto_adjust=False, progress=False)
    print(almanac_dashboard(spy, 2024))


if __name__ == "__main__":
    main()
