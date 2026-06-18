"""
data/track_skew.py — Append today's skew summary to the running history.

Reads each ticker's signals file and extracts:
    - ATM IV (from the atm_iv column — consistent across all rows per expiry)
    - Mean put skew (average iv_vs_atm for flagged puts)
    - Mean call skew (average iv_vs_atm for flagged calls)
    - Count of flagged contracts

Appends one row per ticker per day to data/skew_history.csv.
If the file doesn't exist yet it is created with headers.
If today's date already exists for a ticker (re-run scenario) it is overwritten.

Usage
-----
    PYTHONPATH=. python3 -m data.track_skew
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import pandas as pd


TICKERS = ["SPY", "AAPL", "QQQ", "TSLA", "NVDA", "GLD"]

SIGNAL_FILES = {
    "SPY":  "data/spy_signals.csv",
    "AAPL": "data/aapl_signals.csv",
    "QQQ":  "data/qqq_signals.csv",
    "TSLA": "data/tsla_signals.csv",
    "NVDA": "data/nvda_signals.csv",
    "GLD":  "data/gld_signals.csv",
}

HISTORY_PATH = "data/skew_history.csv"

HISTORY_COLUMNS = [
    "date", "ticker",
    "atm_iv",           # median ATM IV across all flagged expiries
    "put_skew_mean",    # mean iv_vs_atm for flagged puts
    "call_skew_mean",   # mean iv_vs_atm for flagged calls
    "put_skew_max",     # max iv_vs_atm for flagged puts
    "call_skew_max",    # max iv_vs_atm for flagged calls
    "n_puts",           # number of flagged put contracts
    "n_calls",          # number of flagged call contracts
]


def summarise(signals_path: str, ticker: str, date: str) -> dict | None:
    """Compute daily skew summary from a signals CSV. Returns None if file missing."""
    path = Path(signals_path)
    if not path.exists():
        print(f"  {ticker}: signals file not found at {signals_path}, skipping")
        return None

    df = pd.read_csv(path)
    if df.empty:
        print(f"  {ticker}: no flagged contracts today")
        return {
            "date": date, "ticker": ticker,
            "atm_iv": None,
            "put_skew_mean": None, "call_skew_mean": None,
            "put_skew_max": None,  "call_skew_max": None,
            "n_puts": 0, "n_calls": 0,
        }

    puts  = df[df["kind"] == "put"]
    calls = df[df["kind"] == "call"]

    return {
        "date":           date,
        "ticker":         ticker,
        "atm_iv":         round(df["atm_iv"].median(), 2),
        "put_skew_mean":  round(puts["iv_vs_atm"].mean(), 2)  if not puts.empty  else None,
        "call_skew_mean": round(calls["iv_vs_atm"].mean(), 2) if not calls.empty else None,
        "put_skew_max":   round(puts["iv_vs_atm"].max(), 2)   if not puts.empty  else None,
        "call_skew_max":  round(calls["iv_vs_atm"].max(), 2)  if not calls.empty else None,
        "n_puts":         len(puts),
        "n_calls":        len(calls),
    }


def update_history(rows: list[dict], history_path: str) -> pd.DataFrame:
    """Append new rows to history, overwriting any existing rows for the same date/ticker."""
    path = Path(history_path)

    if path.exists():
        history = pd.read_csv(path)
    else:
        history = pd.DataFrame(columns=HISTORY_COLUMNS)

    new = pd.DataFrame(rows)

    # Remove any existing rows for today's date (handles re-runs)
    if not history.empty and not new.empty:
        today = new["date"].iloc[0]
        history = history[history["date"] != today]

    history = pd.concat([history, new], ignore_index=True)
    history = history.sort_values(["date", "ticker"]).reset_index(drop=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(path, index=False)
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Append daily skew summary to history.")
    parser.add_argument("--date", default=datetime.date.today().isoformat(),
                        help="Date to record (default: today)")
    parser.add_argument("--history", default=HISTORY_PATH)
    args = parser.parse_args()

    print(f"Recording skew history for {args.date} ...")
    rows = []
    for ticker in TICKERS:
        row = summarise(SIGNAL_FILES[ticker], ticker, args.date)
        if row is not None:
            rows.append(row)
            print(f"  {ticker}: ATM IV={row['atm_iv']}%, "
                  f"put skew={row['put_skew_mean']}pp, "
                  f"call skew={row['call_skew_mean']}pp, "
                  f"flagged={row['n_puts']}p/{row['n_calls']}c")

    if rows:
        history = update_history(rows, args.history)
        print(f"\nHistory updated: {len(history)} rows total → {args.history}")
    else:
        print("No data to record.")
