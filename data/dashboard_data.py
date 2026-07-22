"""
data/dashboard_data.py — Build dashboard.json for the live vol dashboard.

Combines three real data sources into one JSON file consumed by the static
GitHub Pages dashboard at docs/index.html:

- Live price + % change vs previous close (yfinance)
- 30-day historical volatility, today vs yesterday (single yfinance fetch,
  computed at two trailing window endpoints so no separate history file
  is needed for this field)
- ATM IV and dominant skew, today vs the most recent prior day
  (data/skew_history.csv, written daily by data/track_skew.py)

Usage
-----
    PYTHONPATH=. python3 -m data.dashboard_data

Output
------
    data/dashboard.json
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

TICKERS: list[tuple[str, str]] = [
    ("SPY",  "S&P 500 ETF"),
    ("QQQ",  "Nasdaq-100 ETF"),
    ("AAPL", "Apple Inc."),
    ("TSLA", "Tesla Inc."),
    ("NVDA", "Nvidia Corp."),
    ("GLD",  "Gold ETF"),
]

SKEW_HISTORY_PATH = "data/skew_history.csv"
OUTPUT_PATH = "data/dashboard.json"


# ---------------------------------------------------------------------------
# Price
# ---------------------------------------------------------------------------

def price_and_change(symbol: str) -> tuple[float | None, float | None]:
    """Current price and % change vs the previous close."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")
        if len(hist) < 2:
            return None, None
        last = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2])
        if not (np.isfinite(last) and np.isfinite(prev)) or prev == 0:
            return None, None
        change_pct = (last / prev - 1.0) * 100
        return round(last, 2), round(change_pct, 2)
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Historical volatility — today vs yesterday from a single fetch
# ---------------------------------------------------------------------------

def hist_vol_today_and_yesterday(
    symbol: str, window: int = 30
) -> tuple[float | None, float | None]:
    """30-day realised vol as of today's close and as of yesterday's close.

    Both figures come from one price history fetch: 'today' uses the most
    recent `window` daily returns, 'yesterday' uses the same window shifted
    back by one day. This gives a genuine day-over-day vol comparison
    without needing a separate stored history for this field.
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=f"{window + 15}d")
        closes = hist["Close"]
        if len(closes) < window + 2:
            return None, None
        log_ret = np.log(closes / closes.shift(1)).dropna()

        vol_today = float(log_ret.tail(window).std() * np.sqrt(252) * 100)
        vol_yesterday = float(
            log_ret.iloc[-(window + 1):-1].std() * np.sqrt(252) * 100
        )
        return round(vol_today, 2), round(vol_yesterday, 2)
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# ATM IV and dominant skew from skew_history.csv
# ---------------------------------------------------------------------------

def atm_and_skew_today_and_prior(symbol: str) -> dict:
    """Pull today's and the most recent prior day's ATM IV / skew."""
    result = {
        "atmIv": None, "atmIvChange": None,
        "skewType": "put", "skewValue": None, "skewChange": None,
    }

    path = Path(SKEW_HISTORY_PATH)
    if not path.exists():
        return result

    hist = pd.read_csv(path, parse_dates=["date"])
    sub = hist[hist["ticker"] == symbol].sort_values("date")
    sub = sub.dropna(subset=["atm_iv"])
    if sub.empty:
        return result

    today_row = sub.iloc[-1]
    prior_row = sub.iloc[-2] if len(sub) >= 2 else None

    result["atmIv"] = float(today_row["atm_iv"])
    if prior_row is not None and pd.notna(prior_row.get("atm_iv")):
        result["atmIvChange"] = round(
            float(today_row["atm_iv"]) - float(prior_row["atm_iv"]), 2
        )
    else:
        result["atmIvChange"] = 0.0

    put_mean = today_row.get("put_skew_mean")
    call_mean = today_row.get("call_skew_mean")
    put_mean = float(put_mean) if pd.notna(put_mean) else 0.0
    call_mean = float(call_mean) if pd.notna(call_mean) else 0.0

    if put_mean >= call_mean:
        result["skewType"] = "put"
        result["skewValue"] = round(put_mean, 2)
        prior_val = (
            float(prior_row["put_skew_mean"])
            if prior_row is not None and pd.notna(prior_row.get("put_skew_mean"))
            else None
        )
    else:
        result["skewType"] = "call"
        result["skewValue"] = round(call_mean, 2)
        prior_val = (
            float(prior_row["call_skew_mean"])
            if prior_row is not None and pd.notna(prior_row.get("call_skew_mean"))
            else None
        )

    result["skewChange"] = (
        round(result["skewValue"] - prior_val, 2) if prior_val is not None else 0.0
    )
    return result


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _clean(value):
    """Convert NaN/inf to None so the output is always valid JSON."""
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def build_dashboard() -> dict:
    tickers_out = []
    for symbol, name in TICKERS:
        price, change_pct = price_and_change(symbol)
        vol_today, vol_yesterday = hist_vol_today_and_yesterday(symbol)
        hist_vol_change = (
            round(vol_today - vol_yesterday, 2)
            if vol_today is not None and vol_yesterday is not None
            else None
        )
        skew = atm_and_skew_today_and_prior(symbol)

        tickers_out.append({
            "symbol": symbol,
            "name": name,
            "price": _clean(price),
            "changePct": _clean(change_pct),
            "atmIv": _clean(skew["atmIv"]),
            "atmIvChange": _clean(skew["atmIvChange"]),
            "skewType": skew["skewType"],
            "skewValue": _clean(skew["skewValue"]),
            "skewChange": _clean(skew["skewChange"]),
            "histVol": _clean(vol_today),
            "histVolChange": _clean(hist_vol_change),
        })

    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "tickers": tickers_out,
    }


if __name__ == "__main__":
    data = build_dashboard()
    out = Path(OUTPUT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {OUTPUT_PATH}")
    for t in data["tickers"]:
        print(
            f"  {t['symbol']}: price={t['price']} "
            f"atmIv={t['atmIv']} skew={t['skewType']}/{t['skewValue']}"
        )
