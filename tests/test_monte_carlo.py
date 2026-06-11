import numpy as np
import pytest

from pricer import monte_carlo as mc
from pricer.black_scholes import price as bs_price
from pricer.models import OptionParams

REF = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
TOLS = {"loose": 0.20, "tight": 0.05}


@pytest.mark.parametrize("kind", ["call", "put"])
def test_price_close_to_bs(kind):
    result = mc.price(REF, kind, n=500_000, seed=0)
    bs = bs_price(REF, kind)
    assert abs(result["price"] - bs) < TOLS["tight"]


@pytest.mark.parametrize("kind", ["call", "put"])
def test_bs_inside_confidence_interval(kind):
    bs = bs_price(REF, kind)
    hits = sum(
        r["ci_low"] <= bs <= r["ci_high"]
        for seed in range(20)
        for r in [mc.price(REF, kind, n=50_000, seed=seed)]
    )
    assert hits >= 17


def test_put_call_parity():
    c = mc.price(REF, "call", n=500_000, seed=1)["price"]
    p = mc.price(REF, "put",  n=500_000, seed=1)["price"]
    rhs = REF.S * np.exp(-REF.q * REF.T) - REF.K * np.exp(-REF.r * REF.T)
    assert abs((c - p) - rhs) < 0.10


def test_antithetic_reduces_se():
    plain_ses = [mc.price(REF, "call", n=50_000, antithetic=False, control_variate=False, seed=s)["se"] for s in range(5)]
    anti_ses  = [mc.price(REF, "call", n=50_000, antithetic=True,  control_variate=False, seed=s)["se"] for s in range(5)]
    assert np.mean(anti_ses) < np.mean(plain_ses)


def test_control_variate_reduces_se():
    plain = mc.price(REF, "call", n=50_000, antithetic=False, control_variate=False, seed=3)
    cv    = mc.price(REF, "call", n=50_000, antithetic=False, control_variate=True,  seed=3)
    assert cv["se"] < plain["se"]

def test_convergence_output_shape():
    sizes = [1_000, 10_000, 100_000]
    results = mc.convergence(REF, sample_sizes=sizes)
    assert len(results) == 3
    assert all(r["error"] >= 0 for r in results)
    assert results[-1]["error"] < results[0]["error"]


def test_convergence_default_sample_sizes():
    rows = mc.convergence(REF)
    assert len(rows) == 8
    assert rows[0]["n"] == 100

def test_invalid_kind_raises():
    with pytest.raises(ValueError, match="kind"):
        mc.price(REF, kind="straddle")
