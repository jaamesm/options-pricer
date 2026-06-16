"""
data/scanner.py — Mispricing scanner.

For each expiry in the chain, computes the ATM implied volatility and flags
contracts where IV deviates significantly from the ATM IV for that expiry.
This detects genuine skew anomalies rather than the permanent volatility risk
premium (which causes market IV to always exceed historical vol).

Also reports each contract's IV vs 30-day historical vol for context.

Usage
-----
    PYTHONPATH=. python3 -m data.scanner --ticker SPY --threshold 5.0

Output
------
    data/signals.csv  — full flagged results
    data/signals.md   — markdown summary split by calls/puts
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
    """Annualised close-to-close historical volatility over `window` trading days."""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=f"{window + 10}d")
    if len(hist) < window:
        raise ValueError(f"Insufficient price history for {symbol}")
    log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    daily_vol = log_returns.tail(window).std()
    return float(daily_vol * np.sqrt(252))


# ---------------------------------------------------------------------------
# IV solver — safe wrapper
# ---------------------------------------------------------------------------

def _solve_iv(row: pd.Series, kind: str) -> float | None:
    """Solve IV for a single contract row. Returns None on failure."""
    try:
        p = OptionParams(
            S=float(row["S"]),
            K=float(row["strike"]),
            T=float(row["expiry"]),
            r=float(row["r"]),
            sigma=0.2,
            q=float(row.get("q", 0.0)),
        )
        return iv.solve(p, float(row["mid"]), kind=kind)
    except (ValueError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

def scan(
    symbol: str,
    chain_path: str | None = None,
    threshold: float = 5.0,
    min_volume: int = 10,
    min_mid: float = 0.50,
    max_expiry_years: float = 1.0,
    moneyness_band: float = 0.15,
) -> pd.DataFrame:
    """Scan for IV anomalies relative to the ATM IV for each expiry.

    Parameters
    ----------
    symbol          : ticker symbol
    chain_path      : path to existing chain CSV; fetches live data if None
    threshold       : minimum |IV - ATM IV| in percentage points to flag
    min_volume      : drop contracts with volume below this
    min_mid         : drop contracts with mid price below this (filters pennies)
    max_expiry_years: only consider expiries within this range
    moneyness_band  : only consider strikes within this fraction of spot
                      e.g. 0.15 → strikes within ±15% of spot

    Returns
    -------
    pd.DataFrame of flagged contracts only, sorted by expiry then kind then strike
    """
    # --- Load chain
    if chain_path and Path(chain_path).exists():
        chain = pd.read_csv(chain_path)
    else:
        from data.fetch import fetch_chain
        chain = fetch_chain(symbol, max_expiry_years=max_expiry_years,
                            min_volume=min_volume)

    # --- Filters
    chain = chain.dropna(subset=["mid", "strike", "expiry", "S", "r", "q"])
    chain = chain[chain["expiry"] > 0].copy()
    chain = chain[chain["expiry"] >= 14/365].copy()   # exclude < 2 weeks
    chain = chain[chain["expiry"] <= max_expiry_years].copy()
    chain = chain[chain["volume"] >= min_volume].copy()
    chain = chain[chain["mid"] >= min_mid].copy()
    chain = chain[abs(chain["strike"] / chain["S"] - 1) <= moneyness_band].copy()
    chain = chain.reset_index(drop=True)

    if chain.empty:
        return pd.DataFrame()

    # --- Historical vol (once for the whole symbol)
    hist_vol_val = historical_vol(symbol)

    # --- Per-expiry, per-kind processing
    results = []

    for expiry_date, exp_group in chain.groupby("expiry_date"):
        T = float(exp_group["expiry"].iloc[0])
        S = float(exp_group["S"].iloc[0])

        for kind in ("call", "put"):
            kind_group = exp_group[exp_group["kind"] == kind].copy()
            if len(kind_group) < 2:
                continue

            # Solve IV for every contract in this expiry/kind slice
            kind_group["_iv"] = kind_group.apply(
                lambda row: _solve_iv(row, kind), axis=1
            )
            kind_group = kind_group.dropna(subset=["_iv"])
            if kind_group.empty:
                continue

            # ATM IV: strike closest to spot
            atm_idx = (kind_group["strike"] - S).abs().idxmin()
            atm_iv_val = float(kind_group.loc[atm_idx, "_iv"])

            # Flag contracts where IV deviates from ATM IV by >= threshold pp
            for _, row in kind_group.iterrows():
                market_iv = float(row["_iv"])
                iv_vs_atm  = (market_iv - atm_iv_val) * 100
                iv_vs_hist = (market_iv - hist_vol_val) * 100

                if abs(iv_vs_atm) < threshold:
                    continue

                signal = (
                    "IV elevated vs ATM skew" if iv_vs_atm > 0
                    else "IV depressed vs ATM skew"
                )

                results.append({
                    "ticker":      symbol.upper(),
                    "kind":        kind,
                    "strike":      float(row["strike"]),
                    "expiry_date": expiry_date,
                    "expiry":      round(T, 4),
                    "mid":         round(float(row["mid"]), 3),
                    "market_iv":   round(market_iv * 100, 2),
                    "atm_iv":      round(atm_iv_val * 100, 2),
                    "hist_vol":    round(hist_vol_val * 100, 2),
                    "iv_vs_atm":   round(iv_vs_atm, 2),
                    "iv_vs_hist":  round(iv_vs_hist, 2),
                    "signal":      signal,
                })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values(["expiry_date", "kind", "strike"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def write_csv(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_markdown(
    df: pd.DataFrame,
    path: str,
    symbol: str,
    threshold: float,
    hist_vol: float,
) -> None:
    today = datetime.date.today().isoformat()
    lines = [
        f"# Options Mispricing Signal Report — {symbol}",
        "",
        f"**Generated:** {today}  ",
        f"**Method:** IV vs ATM IV per expiry (skew anomaly detection)  ",
        f"**Threshold:** ±{threshold} percentage points vs ATM IV  ",
        f"**Historical Vol (30d):** {hist_vol:.2f}%  ",
        "",
    ]

    if df.empty:
        lines.append("No contracts exceeded the threshold today.")
    else:
        for kind in ("call", "put"):
            subset = df[df["kind"] == kind]
            if subset.empty:
                continue
            lines.append(f"## {kind.capitalize()}s ({len(subset)} flagged)")
            lines.append("")
            lines.append("| Strike | Expiry | Mid | Market IV | ATM IV | IV vs ATM | Hist Vol | Signal |")
            lines.append("|--------|--------|-----|-----------|--------|-----------|----------|--------|")
            for _, row in subset.iterrows():
                vs_atm_str = f"+{row['iv_vs_atm']:.2f}%" if row["iv_vs_atm"] > 0 \
                             else f"{row['iv_vs_atm']:.2f}%"
                lines.append(
                    f"| {row['strike']:.0f} | {row['expiry_date']} | "
                    f"{row['mid']:.2f} | {row['market_iv']:.2f}% | "
                    f"{row['atm_iv']:.2f}% | {vs_atm_str} | "
                    f"{row['hist_vol']:.2f}% | {row['signal']} |"
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
    p.add_argument("--ticker",         default="SPY")
    p.add_argument("--chain",          default=None)
    p.add_argument("--threshold",      type=float, default=5.0,
                   help="IV vs ATM IV threshold in pp (default: 5.0)")
    p.add_argument("--min-volume",     type=int,   default=10)
    p.add_argument("--min-mid",        type=float, default=0.50)
    p.add_argument("--moneyness-band", type=float, default=0.15,
                   help="Max |strike/spot - 1| to include (default: 0.15)")
    p.add_argument("--csv-out",        default="data/signals.csv")
    p.add_argument("--md-out",         default="data/signals.md")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    print(f"Scanning {args.ticker} (threshold: ±{args.threshold}pp vs ATM IV) ...")

    df = scan(
        args.ticker,
        chain_path=args.chain,
        threshold=args.threshold,
        min_volume=args.min_volume,
        min_mid=args.min_mid,
        moneyness_band=args.moneyness_band,
    )

    hist_vol_pct = historical_vol(args.ticker) * 100

    if df.empty:
        print("No contracts flagged — try lowering --threshold.")
    else:
        print(f"Flagged: {len(df):,} contracts")
        print(f"  Calls: {len(df[df['kind']=='call']):,}")
        print(f"  Puts:  {len(df[df['kind']=='put']):,}")
        print(f"Historical vol (30d): {hist_vol_pct:.2f}%")
        print()
        print(df[["kind","strike","expiry_date","market_iv","atm_iv","iv_vs_atm","signal"]].to_string(index=False))

    write_csv(df, args.csv_out)
    write_markdown(df, args.md_out, args.ticker, args.threshold, hist_vol_pct)
    print(f"\nSaved: {args.csv_out}")
    print(f"Saved: {args.md_out}")
