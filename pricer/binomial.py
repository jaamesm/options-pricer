"""
Cox-Ross-Rubinstein (CRR) binomial tree pricer.

Supports European and American exercise.  The tree is built in-place using a
single array of length (N+1), updated backwards from expiry to t=0, which
keeps memory O(N) rather than O(N²).

Reference:
    Cox, J.C., Ross, S.A., Rubinstein, M. (1979).
    "Option pricing: A simplified approach."
    Journal of Financial Economics, 7(3), 229-263.
"""

from __future__ import annotations

import numpy as np

from pricer.black_scholes import price as bs_price
from pricer.models import OptionParams

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _crr_params(p: OptionParams, n: int) -> tuple[float, float, float, float]:
    """Return (dt, u, d, q) for CRR parameterisation.

    q is the risk-neutral up-probability; under q the discounted stock price
    is a martingale.
    """
    dt = p.T / n
    u  = np.exp(p.sigma * np.sqrt(dt))
    d  = 1.0 / u                              # CRR symmetry
    disc = np.exp(-p.r * dt)
    # risk-neutral probability of an up move (with continuous dividend yield)
    q_up = (np.exp((p.r - p.q) * dt) - d) / (u - d)
    return dt, u, d, q_up, disc


def _payoff(S: np.ndarray, K: float, kind: str) -> np.ndarray:
    if kind == "call":
        return np.maximum(S - K, 0.0)
    return np.maximum(K - S, 0.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def price(
    p: OptionParams,
    kind: str = "call",
    n: int = 500,
    exercise: str = "european",
) -> float:
    """Price a European or American option with a CRR binomial tree.

    Parameters
    ----------
    p        : OptionParams dataclass
    kind     : 'call' or 'put'
    n        : number of time steps (default 500 gives <0.01 error vs BS)
    exercise : 'european' (default) or 'american'

    Returns
    -------
    float — option price
    """
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got '{kind}'")
    if exercise not in ("european", "american"):
        raise ValueError(f"exercise must be 'european' or 'american', got '{exercise}'")
    if n < 1:
        raise ValueError("n must be >= 1")

    dt, u, d, q_up, disc = _crr_params(p, n)
    q_dn = 1.0 - q_up

    # Terminal stock prices S_0 * u^j * d^(N-j)  for j = 0..N
    j = np.arange(n + 1)
    S_T = p.S * (u ** j) * (d ** (n - j))
    V = _payoff(S_T, p.K, kind)

    # Backward induction
    for _ in range(n):
        V = disc * (q_up * V[1:] + q_dn * V[:-1])
        if exercise == "american":
            j = np.arange(len(V))
            S_now = p.S * (u ** j) * (d ** (n - 1 - _ - j))  # noqa: SIM118
            # recompute intrinsic at this layer
            intrinsic = _payoff(S_now, p.K, kind)
            V = np.maximum(V, intrinsic)

    return float(V[0])


def convergence(
    p: OptionParams,
    kind: str = "call",
    exercise: str = "european",
    step_sizes: list[int] | None = None,
) -> list[dict]:
    """Return a convergence table for increasing tree depths.

    Each row contains n, price, bs_price (European only), and error.
    For American options bs_price is set to the N=2000 tree price as
    a pseudo-benchmark (no closed-form exists).
    """
    if step_sizes is None:
        step_sizes = [10, 25, 50, 100, 200, 500, 1_000]

    if exercise == "european":
        benchmark = bs_price(p, kind)
    else:
        # Use a large tree as reference
        benchmark = price(p, kind, n=2_000, exercise="american")

    results = []
    for n in step_sizes:
        v = price(p, kind, n=n, exercise=exercise)
        results.append({
            "n": n,
            "price": v,
            "benchmark": benchmark,
            "error": abs(v - benchmark),
        })
    return results
