"""
Crank-Nicolson finite-difference solver for the Black-Scholes PDE.

The solver works on a uniform grid in log-moneyness x = ln(S/K), which
gives a constant-coefficient PDE and avoids grid clustering near the origin.

PDE (after substitution x = ln S, dropping dividend for clarity):

    dV/dt + ½σ²(d²V/dx²) + (r - q - ½σ²)(dV/dx) - rV = 0

Boundary conditions
-------------------
Call:
    V(x → -∞, t) ≈ 0
    V(x → +∞, t) ≈ S - K e^{-r(T-t)}    (deep ITM)

Put:
    V(x → -∞, t) ≈ K e^{-r(T-t)} - S    (deep ITM)
    V(x → +∞, t) ≈ 0

For American puts an early-exercise constraint V ≥ max(K - S, 0) is enforced
at each time step via a projected SOR (PSOR) iteration.

References
----------
Wilmott, P., Howison, S., & Dewynne, J. (1995).
    The Mathematics of Financial Derivatives.
Duffy, D.J. (2006).
    Finite Difference Methods in Financial Engineering.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

from pricer.black_scholes import price as bs_price
from pricer.models import OptionParams

# ---------------------------------------------------------------------------
# Tri-diagonal helpers
# ---------------------------------------------------------------------------

def _solve_tridiagonal(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve A x = rhs where A is tridiagonal (scipy banded storage)."""
    ab = np.zeros((3, len(diag)))
    ab[0, 1:] = upper[:-1]   # superdiagonal (offset +1)
    ab[1, :]  = diag
    ab[2, :-1] = lower[1:]   # subdiagonal  (offset -1)
    return solve_banded((1, 1), ab, rhs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def price(
    p: OptionParams,
    kind: str = "call",
    exercise: str = "european",
    m: int = 200,
    n: int = 200,
    x_width: float = 4.0,
) -> float:
    """Price a European or American option via Crank-Nicolson.

    Parameters
    ----------
    p        : OptionParams dataclass
    kind     : 'call' or 'put'
    exercise : 'european' or 'american'
    m        : number of interior spatial grid points (default 200)
    n        : number of time steps               (default 200)
    x_width  : half-width of the log-price grid in units of σ√T
               (default 4.0 → ±4 standard deviations)

    Returns
    -------
    float — option price interpolated at S = p.S
    """
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got '{kind}'")
    if exercise not in ("european", "american"):
        raise ValueError(f"exercise must be 'european' or 'american', got '{exercise}'")

    sigma, r, q, K, S, T = p.sigma, p.r, p.q, p.K, p.S, p.T

    # --- Spatial grid in log-price (not log-moneyness, so we keep K separate)
    x_max = x_width * sigma * np.sqrt(T)
    x_min = -x_max
    x = np.linspace(x_min, x_max, m + 2)  # includes boundary nodes
    dx = x[1] - x[0]

    # Stock price at each node
    S_grid = S * np.exp(x)   # x=0 → S_grid = S (ATM)

    # --- Terminal condition (payoff at t = T)
    if kind == "call":
        V = np.maximum(S_grid - K, 0.0)
    else:
        V = np.maximum(K - S_grid, 0.0)

    # --- PDE coefficients (constant on log-price grid)
    alpha = 0.5 * sigma**2
    beta  = r - q - 0.5 * sigma**2   # drift in log-price

    dt = T / n

    # Interior indices
    m_int = m  # number of interior points

    # Crank-Nicolson θ = 0.5
    # Implicit side matrix A, explicit side matrix B
    # A V^{k+1} = B V^k  (time-marching backwards is equivalent to)
    # Rewrite as: march from T → 0 with V known at T

    lam_a = alpha * dt / dx**2    # diffusion coefficient
    lam_b = beta  * dt / (2*dx)   # advection coefficient (centred)

    # Coefficients for interior nodes
    a_sub  = -(0.5 * lam_a - 0.5 * lam_b)   # V_{i-1}
    a_diag =  (1.0 + lam_a + 0.5 * r * dt)   # V_i
    a_sup  = -(0.5 * lam_a + 0.5 * lam_b)   # V_{i+1}

    b_sub  =  (0.5 * lam_a - 0.5 * lam_b)
    b_diag =  (1.0 - lam_a - 0.5 * r * dt)
    b_sup  =  (0.5 * lam_a + 0.5 * lam_b)

    # Build constant tri-diagonal arrays for interior
    lower_A = np.full(m_int, a_sub)
    diag_A  = np.full(m_int, a_diag)
    upper_A = np.full(m_int, a_sup)

    # Time loop: step backwards from T to 0
    for k in range(n):
        tau = (k + 1) * dt   # time remaining at the new level

        # --- Boundary conditions at current level (tau = k*dt)
        if kind == "call":
            bc_left  = 0.0
            bc_right = S_grid[-1] - K * np.exp(-r * tau)
            bc_right = max(bc_right, 0.0)
        else:
            bc_left  = K * np.exp(-r * tau) - S_grid[0]
            bc_left  = max(bc_left, 0.0)
            bc_right = 0.0

        # --- Build RHS = B * V_interior + boundary corrections
        V_int = V[1:-1]
        rhs = (
            b_diag * V_int
            + b_sup  * np.roll(V_int, -1)  # V_{i+1}
            + b_sub  * np.roll(V_int,  1)  # V_{i-1}
        )
        # Fix endpoints (roll wraps incorrectly at edges)
        rhs[0]  = b_diag * V_int[0]  + b_sup * V_int[1]   + b_sub * V[0]
        rhs[-1] = b_diag * V_int[-1] + b_sub * V_int[-2]  + b_sup * V[-1]

        # Boundary contributions to A side
        rhs[0]  += -a_sub  * bc_left
        rhs[-1] += -a_sup  * bc_right

        # --- Solve A V^{new} = rhs
        V_new_int = _solve_tridiagonal(lower_A, diag_A, upper_A, rhs)

        # --- Early-exercise constraint (American)
        if exercise == "american":
            intrinsic = np.maximum(K - S_grid[1:-1], 0.0) if kind == "put" \
                   else np.maximum(S_grid[1:-1] - K, 0.0)
            V_new_int = np.maximum(V_new_int, intrinsic)

        V = np.empty(m + 2)
        V[0]    = bc_left
        V[-1]   = bc_right
        V[1:-1] = V_new_int

    # --- Interpolate at S = p.S  (x = 0 on our grid)
    # Find the two nodes straddling x=0
    x_target = 0.0   # ln(S/S) = 0
    idx = np.searchsorted(x, x_target) - 1
    idx = np.clip(idx, 0, m)
    if idx + 1 > m + 1:
        return float(V[idx])
    frac = (x_target - x[idx]) / (x[idx + 1] - x[idx])
    return float(V[idx] * (1.0 - frac) + V[idx + 1] * frac)


def convergence(
    p: OptionParams,
    kind: str = "call",
    exercise: str = "european",
    grid_sizes: list[tuple[int, int]] | None = None,
) -> list[dict]:
    """Convergence table for increasing (m, n) grid pairs.

    For European options the benchmark is the analytic BS price.
    For American options a fine-grid CN price (m=n=500) is used.
    """
    if grid_sizes is None:
        grid_sizes = [(20, 20), (50, 50), (100, 100), (200, 200), (400, 400)]

    if exercise == "european":
        benchmark = bs_price(p, kind)
    else:
        benchmark = price(p, kind, exercise="american", m=500, n=500)

    results = []
    for m, n in grid_sizes:
        v = price(p, kind, exercise=exercise, m=m, n=n)
        results.append({
            "m": m, "n": n,
            "price": v,
            "benchmark": benchmark,
            "error": abs(v - benchmark),
        })
    return results
