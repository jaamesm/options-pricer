"""
Implied volatility solver and volatility surface.

IV Solver
---------
Uses Brent's method (scipy.optimize.brentq) with an initial bracket
[σ_lo, σ_hi] = [1e-6, 10.0].  Brent's method is guaranteed to converge
within the bracket and avoids the instability of Newton-based methods near
zero vega (very deep ITM / very short dated).

Volatility Surface
------------------
fit_surface() accepts a DataFrame with columns:
    strike  (float)   — option strike
    expiry  (float)   — time to expiry in years
    mid     (float)   — option mid price
    kind    (str)     — 'call' or 'put'
    S       (float)   — spot price at observation time
    r       (float)   — risk-free rate
    q       (float)   — dividend yield (default 0)

and returns a DataFrame with an added 'iv' column plus a 2-D interpolator
(scipy.interpolate.RectBivariateSpline or LinearNDInterpolator depending on
grid regularity).

Usage example
-------------
>>> from pricer.models import OptionParams
>>> from pricer import implied_vol as iv
>>> p = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
>>> from pricer.black_scholes import price as bs_price
>>> mkt_price = bs_price(p, 'call')
>>> print(iv.solve(p, mkt_price, 'call'))   # should be ~0.20
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline, griddata
from scipy.optimize import brentq

from pricer.black_scholes import price as bs_price
from pricer.models import OptionParams

# ---------------------------------------------------------------------------
# Single-contract IV solver
# ---------------------------------------------------------------------------

_SIGMA_LO: float = 1e-6
_SIGMA_HI: float = 10.0


def solve(
    p: OptionParams,
    market_price: float,
    kind: str = "call",
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float:
    """Return the implied volatility for one option contract.

    Parameters
    ----------
    p            : OptionParams — all fields except sigma are used; sigma is
                   ignored (it is the unknown being solved for).
    market_price : observed mid price
    kind         : 'call' or 'put'
    tol          : convergence tolerance (default 1e-8)
    max_iter     : maximum Brent iterations

    Returns
    -------
    float — implied volatility σ* such that BS(σ*) = market_price

    Raises
    ------
    ValueError  — if market_price violates no-arbitrage bounds, or if the
                  solver fails to converge.
    """
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got '{kind}'")

    # --- No-arbitrage floor
    discount      = np.exp(-p.r * p.T)
    div_discount  = np.exp(-p.q * p.T)
    if kind == "call":
        lb = max(p.S * div_discount - p.K * discount, 0.0)
        ub = p.S * div_discount
    else:
        lb = max(p.K * discount - p.S * div_discount, 0.0)
        ub = p.K * discount

    if market_price < lb - tol:
        raise ValueError(
            f"market_price={market_price:.6f} violates lower no-arbitrage "
            f"bound of {lb:.6f}"
        )
    if market_price > ub + tol:
        raise ValueError(
            f"market_price={market_price:.6f} violates upper no-arbitrage "
            f"bound of {ub:.6f}"
        )

    # Clamp to valid interior
    market_price = np.clip(market_price, lb + tol, ub - tol)

    def objective(sigma: float) -> float:
        p_trial = OptionParams(
            S=p.S, K=p.K, T=p.T, r=p.r, sigma=sigma, q=p.q
        )
        return bs_price(p_trial, kind) - market_price

    try:
        iv = brentq(
            objective,
            _SIGMA_LO, _SIGMA_HI,
            xtol=tol, maxiter=max_iter,
        )
    except ValueError as exc:
        raise ValueError(
            f"IV solver failed to bracket root for market_price={market_price:.6f}: {exc}"
        ) from exc

    return float(iv)


# ---------------------------------------------------------------------------
# Vectorised solver over a DataFrame
# ---------------------------------------------------------------------------

def solve_chain(
    chain: pd.DataFrame,
    tol: float = 1e-8,
    on_error: str = "nan",
) -> pd.Series:
    """Compute implied vols for every row in an options-chain DataFrame.

    Expected columns: strike, expiry, mid, kind, S, r, q (optional).

    Parameters
    ----------
    chain    : pd.DataFrame with one row per option contract
    tol      : solver tolerance passed to solve()
    on_error : 'nan' (default) to silently return NaN on errors,
               'raise' to propagate exceptions

    Returns
    -------
    pd.Series of implied vols aligned with chain.index
    """
    required = {"strike", "expiry", "mid", "kind", "S", "r"}
    missing = required - set(chain.columns)
    if missing:
        raise ValueError(f"chain is missing columns: {missing}")

    ivs: list[float] = []
    for _, row in chain.iterrows():
        p = OptionParams(
            S=float(row["S"]),
            K=float(row["strike"]),
            T=float(row["expiry"]),
            r=float(row["r"]),
            sigma=0.2,          # initial guess ignored by Brent
            q=float(row.get("q", 0.0)),
        )
        try:
            iv = solve(p, float(row["mid"]), kind=str(row["kind"]), tol=tol)
        except (ValueError, RuntimeError) as exc:
            if on_error == "raise":
                raise
            warnings.warn(
                f"IV solve failed for row index={_}: {exc}", stacklevel=2
            )
            iv = float("nan")
        ivs.append(iv)

    return pd.Series(ivs, index=chain.index, name="iv")


# ---------------------------------------------------------------------------
# Volatility surface
# ---------------------------------------------------------------------------

def fit_surface(
    chain: pd.DataFrame,
    tol: float = 1e-8,
) -> tuple[pd.DataFrame, Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    """Fit an implied-volatility surface from a chain DataFrame.

    Computes IVs row-by-row, drops NaN rows, then builds a 2-D interpolator
    over (strike, expiry) space.

    Returns
    -------
    enriched_chain : DataFrame with an 'iv' column added (NaN rows dropped)
    interpolator   : callable(K, T) → σ(K, T)
                     Uses RectBivariateSpline when the grid is rectangular
                     (unique strikes × unique expiries); otherwise falls back
                     to LinearNDInterpolator (scattered data).
    """
    df = chain.copy()
    df["iv"] = solve_chain(df, tol=tol, on_error="nan")
    df = df.dropna(subset=["iv"]).reset_index(drop=True)

    K_vals = df["strike"].to_numpy(dtype=float)
    T_vals = df["expiry"].to_numpy(dtype=float)
    IV_vals = df["iv"].to_numpy(dtype=float)

    # Check if data lies on a regular (K × T) grid
    unique_K = np.sort(np.unique(K_vals))
    unique_T = np.sort(np.unique(T_vals))
    is_rect = len(unique_K) * len(unique_T) == len(df)

    if is_rect and len(unique_K) >= 2 and len(unique_T) >= 2:
        # Data already lies on a regular grid — reshape directly
        grid = np.full((len(unique_T), len(unique_K)), np.nan)
        K_idx = {k: i for i, k in enumerate(unique_K)}
        T_idx = {t: i for i, t in enumerate(unique_T)}
        for k, t, iv in zip(K_vals, T_vals, IV_vals):
            grid[T_idx[t], K_idx[k]] = iv
        kx = min(3, len(unique_T) - 1)
        ky = min(3, len(unique_K) - 1)
        spline = RectBivariateSpline(unique_T, unique_K, grid, kx=kx, ky=ky)
        interpolator = lambda K, T: spline(T, K, grid=False)  # noqa: E731
    else:
        # Scattered data — interpolate onto a regular grid first, then fit a
        # smooth spline. This avoids the piecewise-linear artefacts of
        # LinearNDInterpolator and gives a differentiable surface.
        n_K = min(len(unique_K), 50)
        n_T = min(len(unique_T), 20)
        reg_K = np.linspace(unique_K.min(), unique_K.max(), n_K)
        reg_T = np.linspace(unique_T.min(), unique_T.max(), n_T)
        KK, TT = np.meshgrid(reg_K, reg_T)

        # griddata fills the regular grid from scattered (K, T, IV) points
        grid = griddata(
            points=np.column_stack([K_vals, T_vals]),
            values=IV_vals,
            xi=np.column_stack([KK.ravel(), TT.ravel()]),
            method="cubic",
        ).reshape(n_T, n_K)

        # Fill any remaining NaNs at the boundary with nearest-neighbour values
        nan_mask = np.isnan(grid)
        if nan_mask.any():
            grid_nn = griddata(
                points=np.column_stack([K_vals, T_vals]),
                values=IV_vals,
                xi=np.column_stack([KK.ravel(), TT.ravel()]),
                method="nearest",
            ).reshape(n_T, n_K)
            grid[nan_mask] = grid_nn[nan_mask]

        # Fit bicubic spline on the regular grid — order capped by grid size
        kx = min(3, n_T - 1)
        ky = min(3, n_K - 1)
        spline = RectBivariateSpline(reg_T, reg_K, grid, kx=kx, ky=ky)
        interpolator = lambda K, T: spline(T, K, grid=False)  # noqa: E731

    return df, interpolator
