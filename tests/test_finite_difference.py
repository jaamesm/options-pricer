"""Tests for pricer/finite_difference.py (Crank-Nicolson PDE solver)."""

import numpy as np
import pytest

from pricer import finite_difference as fd
from pricer.black_scholes import price as bs_price
from pricer.models import OptionParams

REF = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
ATM_CALL_BS = 10.4506


# ---------------------------------------------------------------------------
# European options — accuracy vs analytic BS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["call", "put"])
def test_european_converges_to_bs(kind):
    """m=n=200 CN grid should be within 0.02 of the analytic price."""
    v = fd.price(REF, kind, exercise="european", m=200, n=200)
    bs = bs_price(REF, kind)
    assert abs(v - bs) < 0.02, f"{kind}: CN={v:.4f}, BS={bs:.4f}"


def test_atm_call_known_value():
    v = fd.price(REF, "call", m=200, n=200)
    assert abs(v - ATM_CALL_BS) < 0.02


@pytest.mark.parametrize("kind", ["call", "put"])
def test_european_put_call_parity(kind):
    """C - P must satisfy put-call parity to within 0.05."""
    c = fd.price(REF, "call", m=300, n=300)
    p = fd.price(REF, "put",  m=300, n=300)
    rhs = REF.S * np.exp(-REF.q * REF.T) - REF.K * np.exp(-REF.r * REF.T)
    assert abs((c - p) - rhs) < 0.05


# ---------------------------------------------------------------------------
# American put — early exercise premium
# ---------------------------------------------------------------------------

def test_american_put_ge_european_put():
    eu = fd.price(REF, "put", exercise="european", m=200, n=200)
    am = fd.price(REF, "put", exercise="american", m=200, n=200)
    assert am >= eu - 1e-6


def test_american_put_ge_intrinsic():
    """American put must be worth at least its intrinsic value."""
    intrinsic = max(REF.K - REF.S, 0.0)
    am = fd.price(REF, "put", exercise="american", m=200, n=200)
    assert am >= intrinsic - 1e-6


# ---------------------------------------------------------------------------
# Monotonicity / boundary checks
# ---------------------------------------------------------------------------

def test_price_positive():
    assert fd.price(REF, "call") > 0
    assert fd.price(REF, "put") > 0


def test_call_price_increases_with_sigma():
    lo = fd.price(OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.1), "call")
    hi = fd.price(OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.4), "call")
    assert hi > lo


def test_call_price_increases_with_T():
    lo = fd.price(OptionParams(S=100, K=100, T=0.25, r=0.05, sigma=0.2), "call")
    hi = fd.price(OptionParams(S=100, K=100, T=2.0,  r=0.05, sigma=0.2), "call")
    assert hi > lo


def test_deep_otm_call_near_zero():
    """A very OTM call should price near zero."""
    p = OptionParams(S=50, K=200, T=0.5, r=0.05, sigma=0.2)
    assert fd.price(p, "call", m=200, n=200) < 0.10


def test_deep_itm_call_approaches_forward():
    """Deep ITM call ≈ F - K·e^{-rT} (intrinsic)."""
    p = OptionParams(S=200, K=100, T=1.0, r=0.05, sigma=0.2)
    intrinsic = p.S - p.K * np.exp(-p.r * p.T)
    v = fd.price(p, "call", m=300, n=300)
    assert abs(v - intrinsic) < 1.0   # coarser bound — boundary effect


def test_invalid_kind_raises():
    with pytest.raises(ValueError, match="kind"):
        fd.price(REF, kind="strangle")


def test_invalid_exercise_raises():
    with pytest.raises(ValueError, match="exercise"):
        fd.price(REF, exercise="asian")


# ---------------------------------------------------------------------------
# Convergence table
# ---------------------------------------------------------------------------

def test_convergence_shape():
    rows = fd.convergence(REF, grid_sizes=[(20, 20), (100, 100), (200, 200)])
    assert len(rows) == 3
    assert all("error" in r for r in rows)
    assert rows[-1]["error"] < rows[0]["error"]


def test_convergence_american_put():
    rows = fd.convergence(
        REF, kind="put", exercise="american",
        grid_sizes=[(50, 50), (200, 200)],
    )
    assert all(r["price"] > 0 for r in rows)
