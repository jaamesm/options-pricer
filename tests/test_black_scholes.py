import pytest
import numpy as np
from pricer.black_scholes import OptionParams, price, greeks

# Standard reference option used throughout
REF = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)

def test_put_call_parity():
    """C - P = S*e^(-qT) - K*e^(-rT)"""
    c = price(REF, "call")
    p = price(REF, "put")
    lhs = c - p
    rhs = REF.S * np.exp(-REF.q * REF.T) - REF.K * np.exp(-REF.r * REF.T)
    assert abs(lhs - rhs) < 1e-10

def test_call_price_known_value():
    """Cross-check against a textbook value (Hull, 9th ed.)"""
    assert abs(price(REF, "call") - 10.4506) < 1e-3

def test_deep_itm_call_approaches_forward():
    """Deep ITM call price → S - K*e^(-rT)"""
    p = OptionParams(S=200, K=100, T=1.0, r=0.05, sigma=0.2)
    forward = p.S - p.K * np.exp(-p.r * p.T)
    assert abs(price(p, "call") - forward) < 0.01

def test_delta_bounds():
    g = greeks(REF, "call")
    assert 0 < g["delta"] < 1
    g_put = greeks(REF, "put")
    assert -1 < g_put["delta"] < 0

def test_gamma_positive():
    """Gamma is always positive for both calls and puts."""
    assert greeks(REF, "call")["gamma"] > 0
    assert greeks(REF, "put")["gamma"] > 0

def test_call_put_gamma_equal():
    """Gamma is identical for a call and put with same params."""
    assert abs(greeks(REF, "call")["gamma"] - greeks(REF, "put")["gamma"]) < 1e-12
