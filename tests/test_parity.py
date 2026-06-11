"""
Cross-method put-call parity and price consistency tests.

Put-call parity (continuous dividends):
    C - P = S·e^{-qT} - K·e^{-rT}

All four methods — Black-Scholes, Monte Carlo, Binomial (CRR), and
Crank-Nicolson — are tested against this identity and against each other.
"""

import pytest
import numpy as np

from pricer.models import OptionParams
from pricer.black_scholes import price as bs_price
from pricer import monte_carlo as mc
from pricer import binomial
from pricer import finite_difference as fd

# Reference contract (used throughout the project)
REF = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)

# Tighter tolerance for deterministic methods; looser for MC
_TOL_DETERMINISTIC = 0.05
_TOL_MC            = 0.15

# Convenience
def _pcp_rhs(p: OptionParams) -> float:
    return p.S * np.exp(-p.q * p.T) - p.K * np.exp(-p.r * p.T)


# ---------------------------------------------------------------------------
# Black-Scholes (exact — already in test_black_scholes.py, kept for symmetry)
# ---------------------------------------------------------------------------

def test_parity_black_scholes():
    c = bs_price(REF, "call")
    p = bs_price(REF, "put")
    assert abs((c - p) - _pcp_rhs(REF)) < 1e-10


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def test_parity_monte_carlo():
    c = mc.price(REF, "call", n=500_000, seed=99)["price"]
    p = mc.price(REF, "put",  n=500_000, seed=99)["price"]
    assert abs((c - p) - _pcp_rhs(REF)) < _TOL_MC


# ---------------------------------------------------------------------------
# Binomial (CRR)
# ---------------------------------------------------------------------------

def test_parity_binomial():
    c = binomial.price(REF, "call", n=500)
    p = binomial.price(REF, "put",  n=500)
    assert abs((c - p) - _pcp_rhs(REF)) < _TOL_DETERMINISTIC


# ---------------------------------------------------------------------------
# Crank-Nicolson
# ---------------------------------------------------------------------------

def test_parity_crank_nicolson():
    c = fd.price(REF, "call", m=300, n=300)
    p = fd.price(REF, "put",  m=300, n=300)
    assert abs((c - p) - _pcp_rhs(REF)) < _TOL_DETERMINISTIC


# ---------------------------------------------------------------------------
# Cross-method call price consistency
# ---------------------------------------------------------------------------

ATM_CALL_BS = bs_price(REF, "call")   # ≈ 10.4506

@pytest.mark.parametrize("method,call_price", [
    ("binomial_500",  lambda: binomial.price(REF, "call", n=500)),
    ("cn_200x200",    lambda: fd.price(REF, "call", m=200, n=200)),
    ("mc_500k",       lambda: mc.price(REF, "call", n=500_000, seed=7)["price"]),
])
def test_all_methods_agree_with_bs(method, call_price):
    tol = _TOL_MC if "mc" in method else _TOL_DETERMINISTIC
    v = call_price()
    assert abs(v - ATM_CALL_BS) < tol, (
        f"{method}: price={v:.4f}, bs={ATM_CALL_BS:.4f}, diff={abs(v-ATM_CALL_BS):.4f}"
    )


# ---------------------------------------------------------------------------
# OTM and ITM options — all four methods within 0.10 of BS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("params,kind", [
    (OptionParams(S=100, K=110, T=1.0, r=0.05, sigma=0.2), "call"),  # OTM call
    (OptionParams(S=100, K=90,  T=1.0, r=0.05, sigma=0.2), "put"),   # OTM put
    (OptionParams(S=100, K=90,  T=1.0, r=0.05, sigma=0.2), "call"),  # ITM call
    (OptionParams(S=100, K=110, T=1.0, r=0.05, sigma=0.2), "put"),   # ITM put
])
def test_cross_method_otm_itm(params, kind):
    benchmark = bs_price(params, kind)
    results = {
        "binomial": binomial.price(params, kind, n=500),
        "cn":       fd.price(params, kind, m=200, n=200),
        "mc":       mc.price(params, kind, n=500_000, seed=42)["price"],
    }
    tols = {"binomial": 0.05, "cn": 0.05, "mc": 0.15}
    for name, v in results.items():
        assert abs(v - benchmark) < tols[name], (
            f"{name} {kind} K={params.K}: price={v:.4f}, bs={benchmark:.4f}"
        )


# ---------------------------------------------------------------------------
# With continuous dividend yield
# ---------------------------------------------------------------------------

DIV_REF = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.03)

def test_parity_with_dividends_binomial():
    c = binomial.price(DIV_REF, "call", n=500)
    p = binomial.price(DIV_REF, "put",  n=500)
    assert abs((c - p) - _pcp_rhs(DIV_REF)) < _TOL_DETERMINISTIC


def test_parity_with_dividends_cn():
    c = fd.price(DIV_REF, "call", m=200, n=200)
    p = fd.price(DIV_REF, "put",  m=200, n=200)
    assert abs((c - p) - _pcp_rhs(DIV_REF)) < _TOL_DETERMINISTIC
