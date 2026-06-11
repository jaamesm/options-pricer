from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class OptionParams:
    S: float      # spot price
    K: float      # strike price
    T: float      # time to expiry (years)
    r: float      # risk-free rate
    sigma: float  # volatility
    q: float = 0.0  # continuous dividend yield

def _d1_d2(p: OptionParams) -> tuple[float, float]:
    d1 = (np.log(p.S / p.K) + (p.r - p.q + 0.5 * p.sigma**2) * p.T) / (p.sigma * np.sqrt(p.T))
    d2 = d1 - p.sigma * np.sqrt(p.T)
    return d1, d2

def price(p: OptionParams, kind: str = "call") -> float:
    """Black-Scholes price for a European call or put."""
    d1, d2 = _d1_d2(p)
    discount = np.exp(-p.r * p.T)
    div_discount = np.exp(-p.q * p.T)
    if kind == "call":
        return p.S * div_discount * norm.cdf(d1) - p.K * discount * norm.cdf(d2)
    elif kind == "put":
        return p.K * discount * norm.cdf(-d2) - p.S * div_discount * norm.cdf(-d1)
    raise ValueError(f"kind must be 'call' or 'put', got '{kind}'")

def greeks(p: OptionParams, kind: str = "call") -> dict[str, float]:
    """Returns all five Greeks for a European call or put."""
    d1, d2 = _d1_d2(p)
    sqrt_T = np.sqrt(p.T)
    discount = np.exp(-p.r * p.T)
    div_discount = np.exp(-p.q * p.T)
    n_d1 = norm.pdf(d1)

    # Shared across call and put
    gamma = (div_discount * n_d1) / (p.S * p.sigma * sqrt_T)
    vega  = p.S * div_discount * n_d1 * sqrt_T  # per unit vol (not per 1%)

    if kind == "call":
        delta = div_discount * norm.cdf(d1)
        theta = (
            -(p.S * div_discount * n_d1 * p.sigma) / (2 * sqrt_T)
            - p.r * p.K * discount * norm.cdf(d2)
            + p.q * p.S * div_discount * norm.cdf(d1)
        )
        rho = p.K * p.T * discount * norm.cdf(d2)
    elif kind == "put":
        delta = div_discount * (norm.cdf(d1) - 1)
        theta = (
            -(p.S * div_discount * n_d1 * p.sigma) / (2 * sqrt_T)
            + p.r * p.K * discount * norm.cdf(-d2)
            - p.q * p.S * div_discount * norm.cdf(-d1)
        )
        rho = -p.K * p.T * discount * norm.cdf(-d2)
    else:
        raise ValueError(f"kind must be 'call' or 'put', got '{kind}'")

    return {"delta": delta, "gamma": gamma, "theta": theta / 365,  # per calendar day
            "vega": vega / 100, "rho": rho / 100}  # per 1% move
