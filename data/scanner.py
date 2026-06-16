"""
data/scanner.py — Mispricing scanner.

Compares market implied volatility from the options chain against
30-day historical volatility computed from recent price history.
Flags contracts where the IV premium exceeds a configurable threshold.

Usage
-----
    PYTHONPATH=. python3 -m data.scanner --ticker SPY --threshold 3.0

Output
------
    data/signals.csv  — full results table
    data/signals.md   — markdown summary for GitHub viewing
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from pricer import implied_vol as iv
from pricer.models import OptionParams


# ---------------------------------------------------------------------------
# Historical volatility
# ---------------------------------------------------------------------------

def historical_vol(symbol: str, window: int = 30) -> float:
    """Compute annualised close-to-close historical volatility over `window` days."""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=f"{window + 10}d")
    if len(hist) < window:
        raise ValueError(f"Insufficient price history for {symbol}")
    log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    daily_vol = log_returns.tail(window).std()
    return float(daily_vol * np.sqrt(252))


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

def scan(
    symbol: str,
    chain_path: str | None = None,
    threshold: float = 3.0,
    min_volume: int = 10,
    max_expiry_years: float = 1.0,
) -> pd.DataFrame:
    """Scan an options chain for IV vs historical vol mispricings.

    Parameters
    ----------
    symbol          : ticker symbol
    chain_path      : path to CSV chain file; if None, fetches live data
    threshold       : minimum IV premium (percentage points) to flag
    min_volume      : minimum contract volume to include
    max_expiry_years: only consider near-dated contracts

    Returns
    -------
    pd.DataFrame with columns:
        ticker, kind, strike, expiry_date, expiry, mid, market_iv,
        hist_vol, iv_premium, signal
    """
    # --- Load chain
    if chain_path and Path(chain_path).exists():
        chain = pd.read_csv(chain_path)
    else:
        from data.fetch import fetch_chain
        chain = fetch_chain(symbol, max_expiry_years=max_expiry_years,
                           min_volume=min_volume)

    # --- Filter
    chain = chain[chain["volume"] >= min_volume].copy()
    chain = chain[chain["expiry"] <= max_expiry_years].copy()
    chain = chain.dropna(subset=["mid", "strike", "expiry", "S", "r", "q"])

    # Filter to near-the-money contracts only (moneyness within 20% of spot)
    chain = chain[abs(chain["strike"] / chain["S"] - 1) <= 0.20].copy()

    # --- Historical vol
    hist_vol = historical_vol(symbol)

    # --- Solve IV for each contract
    results = []
    for _, row in chain.iterrows():
        p = OptionParams(
            S=float(row["S"]),
            K=float(row["strike"]),
            T=float(row["expiry"]),
            r=float(row["r"]),
            sigma=0.2,
            q=float(row.get("q", 0.0)),
        )
        try:
            market_iv = iv.solve(p, float(row["mid"]), kind=str(row["kind"]))
        except (ValueError, RuntimeError):
            continue

        iv_premium = (market_iv - hist_vol) * 100   # in percentage points

        if abs(iv_premium) >= threshold:
            signal = "IV elevated — option may be overpriced" if iv_premium > 0 \
                else "IV depressed — option may be underpriced"
        else:
            signal = "—"

        results.append({
            "ticker":       symbol.upper(),
            "kind":         row["kind"],
            "strike":       row["strike"],
            "expiry_date":  row["expiry_date"],
            "expiry":       round(float(row["expiry"]), 4),
            "mid":          round(float(row["mid"]), 3),
            "market_iv":    round(market_iv * 100, 2),
            "hist_vol":     round(hist_vol * 100, 2),
            "iv_premium":   round(iv_premium, 2),
            "signal":       signal,
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df

    df = df.sort_values(["kind", "expiry", "strike"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def write_csv(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_markdown(df: pd.DataFrame, path: str, symbol: str, threshold: float) -> None:
    """Write a markdown report split into calls and puts sections."""
    today = datetime.date.today().isoformat()
    lines = [
        f"# Options Mispricing Signal Report — {symbol}",
        f"",
        f"**Generated:** {today}  ",
        f"**Threshold:** ±{threshold} percentage points vs 30-day historical vol  ",
        f"**Historical Vol (30d):** {df['hist_vol'].iloc[0]:.2f}%  ",
        f"",
    ]

    flagged = df[df["signal"] != "—"]

    if flagged.empty:
        lines.append("No contracts exceeded the threshold today.")
    else:
        for kind in ("call", "put"):
            subset = flagged[flagged["kind"] == kind]
            if subset.empty:
                continue
            lines.append(f"## {kind.capitalize()}s")
            lines.append("")
            lines.append("| Strike | Expiry | Mid | Market IV | Hist Vol | IV Premium | Signal |")
            lines.append("|--------|--------|-----|-----------|----------|------------|--------|")
            for _, row in subset.iterrows():
                premium_str = f"+{row['iv_premium']:.2f}%" if row['iv_premium'] > 0 \
                              else f"{row['iv_premium']:.2f}%"
                lines.append(
                    f"| {row['strike']:.0f} | {row['expiry_date']} | "
                    f"{row['mid']:.3f} | {row['market_iv']:.2f}% | "
                    f"{row['hist_vol']:.2f}% | {premium_str} | {row['signal']} |"
                )
            lines.append("")

    lines.append("---")
    lines.append("*Generated by options-pricer mispricing scanner*")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Options mispricing scanner.")
    p.add_argument("--ticker",    default="SPY")
    p.add_argument("--chain",     default=None,  help="Path to existing chain CSV")
    p.add_argument("--threshold", type=float, default=3.0,
                   help="IV premium threshold in percentage points (default: 3.0)")
    p.add_argument("--min-volume", type=int, default=10)
    p.add_argument("--csv-out",   default="data/signals.csv")
    p.add_argument("--md-out",    default="data/signals.md")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    print(f"Scanning {args.ticker} options chain (threshold: ±{args.threshold}pp)...")

    df = scan(
        args.ticker,
        chain_path=args.chain,
        threshold=args.threshold,
        min_volume=args.min_volume,
    )

    if df.empty:
        print("No results — check chain file or filters.")
    else:
        flagged = df[df["signal"] != "—"]
        print(f"Contracts scanned: {len(df):,}")
        print(f"Flagged:           {len(flagged):,}")
        print(f"  Calls:           {len(flagged[flagged['kind']=='call']):,}")
        print(f"  Puts:            {len(flagged[flagged['kind']=='put']):,}")
        print(f"Historical vol:    {df['hist_vol'].iloc[0]:.2f}%")
        print()
        if not flagged.empty:
            print(flagged[["kind","strike","expiry_date","market_iv","hist_vol","iv_premium","signal"]].to_string(index=False))

        write_csv(df, args.csv_out)
        write_markdown(df, args.md_out, args.ticker, args.threshold)
        print(f"\nSaved: {args.csv_out}")
        print(f"Saved: {args.md_out}")
