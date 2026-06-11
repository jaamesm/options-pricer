"""Tests for pricer/implied_vol.py."""

import numpy as np
import pandas as pd
import pytest

from pricer import implied_vol as iv
from pricer.black_scholes import price as bs_price
from pricer.models import OptionParams

REF = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
ATM_CALL_BS = bs_price(REF, "call")


# ---------------------------------------------------------------------------
# Single-contract solver — round-trip accuracy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("sigma", [0.05, 0.10, 0.20, 0.35, 0.60])
def test_round_trip_accuracy(kind, sigma):
    """IV(BS(σ)) ≈ σ to within 1e-6."""
    p = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=sigma)
    mkt = bs_price(p, kind)
    recovered = iv.solve(p, mkt, kind)
    assert abs(recovered - sigma) < 1e-6, (
        f"{kind} σ={sigma}: recovered={recovered:.8f}"
    )


@pytest.mark.parametrize("kind", ["call", "put"])
def test_round_trip_otm(kind):
    """OTM round-trip: call on K=120, put on K=80."""
    strike = 120 if kind == "call" else 80
    sigma = 0.25
    p = OptionParams(S=100, K=strike, T=0.5, r=0.03, sigma=sigma)
    mkt = bs_price(p, kind)
    recovered = iv.solve(p, mkt, kind)
    assert abs(recovered - sigma) < 1e-6


def test_round_trip_with_dividends():
    sigma = 0.18
    p = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=sigma, q=0.02)
    mkt = bs_price(p, "call")
    recovered = iv.solve(p, mkt, "call")
    assert abs(recovered - sigma) < 1e-6


def test_round_trip_short_dated():
    """Short-dated option (1 week) IV round-trip."""
    sigma = 0.30
    p = OptionParams(S=100, K=100, T=7/365, r=0.05, sigma=sigma)
    mkt = bs_price(p, "call")
    recovered = iv.solve(p, mkt, "call")
    assert abs(recovered - sigma) < 1e-4


# ---------------------------------------------------------------------------
# No-arbitrage bound violations
# ---------------------------------------------------------------------------

def test_price_below_lower_bound_raises():
    """Market price below lower no-arb bound should raise ValueError."""
    with pytest.raises(ValueError, match="lower no-arbitrage"):
        iv.solve(REF, market_price=-1.0, kind="call")


def test_price_above_upper_bound_raises():
    """Market price above S (undiscounted spot) should raise ValueError."""
    with pytest.raises(ValueError, match="upper no-arbitrage"):
        iv.solve(REF, market_price=REF.S * 2, kind="call")


def test_invalid_kind_raises():
    with pytest.raises(ValueError, match="kind"):
        iv.solve(REF, market_price=ATM_CALL_BS, kind="digital")


# ---------------------------------------------------------------------------
# solve_chain — vectorised
# ---------------------------------------------------------------------------

def _make_chain(spot=100.0) -> pd.DataFrame:
    strikes  = [90, 95, 100, 105, 110]
    expiries = [0.25, 0.5, 1.0]
    rows = []
    for T in expiries:
        for K in strikes:
            sigma = 0.20 + 0.05 * abs(K - spot) / spot   # smile
            p = OptionParams(S=spot, K=K, T=T, r=0.05, sigma=sigma)
            rows.append({
                "S": spot, "K": K, "strike": K, "expiry": T,
                "mid": bs_price(p, "call"), "kind": "call",
                "r": 0.05, "q": 0.0,
                "true_iv": sigma,
            })
    return pd.DataFrame(rows)


def test_solve_chain_recovers_smile():
    chain = _make_chain()
    ivs = iv.solve_chain(chain)
    assert len(ivs) == len(chain)
    assert ivs.isna().sum() == 0
    for true_iv, computed_iv in zip(chain["true_iv"], ivs):
        assert abs(computed_iv - true_iv) < 1e-5


def test_solve_chain_bad_row_returns_nan():
    chain = _make_chain()
    # Corrupt one row with an impossible price
    chain.loc[0, "mid"] = -999.0
    ivs = iv.solve_chain(chain, on_error="nan")
    assert np.isnan(ivs.iloc[0])
    assert ivs.iloc[1:].notna().all()


def test_solve_chain_raises_on_error():
    chain = _make_chain()
    chain.loc[0, "mid"] = -999.0
    with pytest.raises(ValueError):
        iv.solve_chain(chain, on_error="raise")


def test_solve_chain_missing_column_raises():
    chain = _make_chain().drop(columns=["r"])
    with pytest.raises(ValueError, match="missing columns"):
        iv.solve_chain(chain)


# ---------------------------------------------------------------------------
# fit_surface — interpolator smoke tests
# ---------------------------------------------------------------------------

def test_fit_surface_returns_df_and_callable():
    chain = _make_chain()
    df, interp = iv.fit_surface(chain)
    assert "iv" in df.columns
    assert callable(interp)


def test_fit_surface_interpolator_atm():
    """Interpolating at an ATM grid point should give back the original IV."""
    chain = _make_chain()
    df, interp = iv.fit_surface(chain)
    # Pick an ATM, T=1.0 point
    row = df[(df["strike"] == 100) & (df["expiry"] == 1.0)].iloc[0]
    predicted = interp(np.array([100.0]), np.array([1.0]))
    assert abs(float(np.asarray(predicted).ravel()[0]) - row["iv"]) < 1e-3


def test_fit_surface_drops_nan_rows():
    chain = _make_chain()
    chain.loc[0, "mid"] = -999.0   # will produce NaN IV
    df, _ = iv.fit_surface(chain)
    assert df["iv"].isna().sum() == 0
