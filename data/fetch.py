"""
data/fetch.py — Pull an options chain from Yahoo Finance via yfinance.

Usage
-----
    python -m data.fetch --ticker SPY --output data/sample_chain.csv
"""

from __future__ import annotations

import argparse
import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


def _get_risk_free_rate(default_r: float = 0.05) -> float:
    try:
        irx = yf.Ticker("^IRX")
        hist = irx.history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1]) / 100.0
    except Exception:
        pass
    warnings.warn(f"Could not fetch ^IRX; using default r={default_r}", stacklevel=2)
    return default_r


def _get_dividend_yield(ticker: yf.Ticker) -> float:
    try:
        info = ticker.info
        q = info.get("trailingAnnualDividendYield") or 0.0
        return float(q)
    except Exception:
        return 0.0


def _get_spot(ticker: yf.Ticker) -> float:
    try:
        return float(ticker.fast_info["last_price"])
    except Exception:
        hist = ticker.history(period="1d")
        if hist.empty:
            raise ValueError("Cannot determine spot price")
        return float(hist["Close"].iloc[-1])


def fetch_chain(
    symbol: str,
    max_expiry_years: float = 1.5,
    min_volume: int = 0,
    default_r: float = 0.05,
) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    S  = _get_spot(ticker)
    r  = _get_risk_free_rate(default_r)
    q  = _get_dividend_yield(ticker)
    today = datetime.date.today()
    cutoff = today + datetime.timedelta(days=int(max_expiry_years * 365))

    expiry_dates = ticker.options
    if not expiry_dates:
        raise ValueError(f"No options data found for {symbol!r}")

    frames = []
    for exp_str in expiry_dates:
        exp_date = datetime.date.fromisoformat(exp_str)
        if exp_date > cutoff or exp_date <= today:
            continue
        T = (exp_date - today).days / 365.0
        try:
            chain = ticker.option_chain(exp_str)
        except Exception as exc:
            warnings.warn(f"Skipping expiry {exp_str}: {exc}", stacklevel=2)
            continue

        for kind, df in (("call", chain.calls), ("put", chain.puts)):
            df = df.copy()
            df["kind"]        = kind
            df["expiry"]      = T
            df["expiry_date"] = exp_str
            df["S"]           = S
            df["r"]           = r
            df["q"]           = q
            df["ticker"]      = symbol.upper()
            frames.append(df)

    if not frames:
        raise ValueError(f"No valid near-dated expiries found for {symbol!r}")

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.rename(columns={
        "lastPrice":         "last_price",
        "impliedVolatility": "implied_volatility_yf",
    })

    raw["bid"] = pd.to_numeric(raw["bid"], errors="coerce").fillna(0.0)
    raw["ask"] = pd.to_numeric(raw["ask"], errors="coerce").fillna(0.0)
    raw["mid"] = (raw["bid"] + raw["ask"]) / 2.0
    raw = raw[raw["mid"] > 0]
    raw = raw[raw["bid"] > 0]
    raw = raw[raw["ask"] > 0]

    if min_volume > 0:
        raw["volume"] = pd.to_numeric(raw["volume"], errors="coerce").fillna(0)
        raw = raw[raw["volume"] >= min_volume]

    keep = [
        "ticker", "kind", "strike", "expiry", "expiry_date",
        "mid", "bid", "ask", "last_price", "volume", "open_interest",
        "implied_volatility_yf", "S", "r", "q",
    ]
    existing = [c for c in keep if c in raw.columns]
    result = raw[existing].reset_index(drop=True)
    result["strike"] = result["strike"].astype(float)
    result["expiry"] = result["expiry"].astype(float)
    return result


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fetch an options chain via yfinance.")
    p.add_argument("--ticker",  default="SPY")
    p.add_argument("--output",  default="data/sample_chain.csv")
    p.add_argument("--max-expiry-years", type=float, default=1.5)
    p.add_argument("--min-volume", type=int, default=10)
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    print(f"Fetching {args.ticker} options chain ...")
    df = fetch_chain(args.ticker, max_expiry_years=args.max_expiry_years, min_volume=args.min_volume)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} contracts -> {out}")
    print(df.head(10).to_string(index=False))
