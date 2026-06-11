"""Tests for pricer/binomial.py (CRR tree)."""

import pytest
import numpy as np

from pricer.models import OptionParams
from pricer.black_scholes import price as bs_price
from pricer import binomial

REF = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
ATM_CALL_BS = 10.4506   # analytic reference


# ---------------------------------------------------------------------------
# European options — convergence to BS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["call", "put"])
def test_european_converges_to_bs(kind):
    """N=500 CRR tree should be within 0.01 of the analytic price."""
    v = binomial.price(REF, kind, n=500, exercise="european")
    bs = bs_price(REF, kind)
    assert abs(v - bs) < 0.01, f"{kind}: binomial={v:.4f}, bs={bs:.4f}"


def test_atm_call_known_value():
    v = binomial.price(REF, "call", n=500)
    assert abs(v - ATM_CALL_BS) < 0.01


@pytest.mark.parametrize("kind", ["call", "put"])
def test_european_put_call_parity(kind):
    """C - P = S·e^{-qT} - K·e^{-rT} to within 0.01."""
    c = binomial.price(REF, "call", n=500)
    p = binomial.price(REF, "put",  n=500)
    rhs = REF.S * np.exp(-REF.q * REF.T) - REF.K * np.exp(-REF.r * REF.T)
    assert abs((c - p) - rhs) < 0.01


# ---------------------------------------------------------------------------
# American options — put premium over European
# ---------------------------------------------------------------------------

def test_american_put_ge_european_put():
    """American put must be worth at least as much as European put."""
    eu = binomial.price(REF, "put", n=500, exercise="european")
    am = binomial.price(REF, "put", n=500, exercise="american")
    assert am >= eu - 1e-9


def test_american_call_nodividend_equals_european():
    """For q=0 the American call equals the European call (no early exercise)."""
    p_nodiv = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.0)
    eu = binomial.price(p_nodiv, "call", n=500, exercise="european")
    am = binomial.price(p_nodiv, "call", n=500, exercise="american")
    assert abs(am - eu) < 0.01


def test_american_put_premium_otm():
    """Deep OTM American put should have negligible early-exercise premium."""
    deep_otm = OptionParams(S=100, K=60, T=1.0, r=0.05, sigma=0.2)
    eu = binomial.price(deep_otm, "put", n=300, exercise="european")
    am = binomial.price(deep_otm, "put", n=300, exercise="american")
    # Premium should be tiny
    assert abs(am - eu) < 0.05


# ---------------------------------------------------------------------------
# Boundary / monotonicity checks
# ---------------------------------------------------------------------------

def test_price_positive():
    assert binomial.price(REF, "call") > 0
    assert binomial.price(REF, "put") > 0


def test_price_increases_with_sigma():
    lo = binomial.price(OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.1), "call")
    hi = binomial.price(OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.4), "call")
    assert hi > lo


def test_price_increases_with_T():
    lo = binomial.price(OptionParams(S=100, K=100, T=0.25, r=0.05, sigma=0.2), "call")
    hi = binomial.price(OptionParams(S=100, K=100, T=2.0,  r=0.05, sigma=0.2), "call")
    assert hi > lo


def test_invalid_kind_raises():
    with pytest.raises(ValueError, match="kind"):
        binomial.price(REF, kind="invalid")


def test_invalid_exercise_raises():
    with pytest.raises(ValueError, match="exercise"):
        binomial.price(REF, exercise="bermudan")


# ---------------------------------------------------------------------------
# Convergence table
# ---------------------------------------------------------------------------

def test_convergence_shape_and_monotone():
    sizes = [10, 50, 200]
    rows = binomial.convergence(REF, step_sizes=sizes)
    assert len(rows) == 3
    # Error should decrease as N increases (not guaranteed monotone, but
    # the largest N should beat the smallest for n≥10)
    assert rows[-1]["error"] < rows[0]["error"]


def test_convergence_american_put():
    rows = binomial.convergence(REF, "put", exercise="american", step_sizes=[50, 200])
    assert all(r["price"] > 0 for r in rows)
    assert rows[0]["benchmark"] > 0
