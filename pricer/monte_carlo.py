import numpy as np
from pricer.models import OptionParams
from pricer.black_scholes import price as bs_price

RNG = np.random.default_rng()

def _simulate_terminal_prices(p: OptionParams, n: int, rng: np.random.Generator) -> np.ndarray:
    """GBM terminal price: S_T = S * exp((r - q - σ²/2)T + σ√T * Z)"""
    Z = rng.standard_normal(n)
    return p.S * np.exp((p.r - p.q - 0.5 * p.sigma**2) * p.T + p.sigma * np.sqrt(p.T) * Z)

def price(
    p: OptionParams,
    kind: str = "call",
    n: int = 100_000,
    antithetic: bool = True,
    control_variate: bool = True,
    seed: int | None = None,
) -> dict[str, float]:
    """
    Monte Carlo pricer for European options.

    Variance reduction:
      - Antithetic variates:  use Z and -Z to halve variance
      - Control variate:      use the BS price of a digital as a known anchor

    Returns estimate, standard error, and 95% CI.
    """
    rng = np.random.default_rng(seed)
    discount = np.exp(-p.r * p.T)

    if antithetic:
        half = n // 2
        Z = rng.standard_normal(half)
        S_plus  = p.S * np.exp((p.r - p.q - 0.5 * p.sigma**2) * p.T + p.sigma * np.sqrt(p.T) *  Z)
        S_minus = p.S * np.exp((p.r - p.q - 0.5 * p.sigma**2) * p.T + p.sigma * np.sqrt(p.T) * -Z)
        S_T = np.concatenate([S_plus, S_minus])
    else:
        S_T = _simulate_terminal_prices(p, n, rng)

    if kind == "call":
        payoffs = np.maximum(S_T - p.K, 0.0)
    elif kind == "put":
        payoffs = np.maximum(p.K - S_T, 0.0)
    else:
        raise ValueError(f"kind must be 'call' or 'put', got '{kind}'")

    if control_variate:
        # Control: indicator payoff 1_{S_T > K}, whose BS price is e^{-rT} * N(d2)
        from scipy.stats import norm
        d2 = (np.log(p.S / p.K) + (p.r - p.q - 0.5 * p.sigma**2) * p.T) / (p.sigma * np.sqrt(p.T))
        cv_true  = discount * norm.cdf(d2 if kind == "call" else -d2)
        cv_sim   = discount * (S_T > p.K if kind == "call" else S_T < p.K).astype(float)
        # Optimal beta
        beta = np.cov(discount * payoffs, cv_sim)[0, 1] / np.var(cv_sim)
        adjusted = discount * payoffs - beta * (cv_sim - cv_true)
    else:
        adjusted = discount * payoffs

    est = adjusted.mean()
    se  = adjusted.std() / np.sqrt(n)

    return {"price": est, "se": se, "ci_low": est - 1.96 * se, "ci_high": est + 1.96 * se}


def convergence(
    p: OptionParams,
    kind: str = "call",
    sample_sizes: list[int] | None = None,
    seed: int = 42,
) -> list[dict]:
    """
    Run the pricer at increasing sample sizes.
    Returns a list of dicts for plotting convergence vs. BS benchmark.
    """
    if sample_sizes is None:
        sample_sizes = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000]
    benchmark = bs_price(p, kind)
    results = []
    for n in sample_sizes:
        r = price(p, kind, n=n, seed=seed)
        results.append({
            "n": n,
            "price": r["price"],
            "ci_low": r["ci_low"],
            "ci_high": r["ci_high"],
            "bs_price": benchmark,
            "error": abs(r["price"] - benchmark),
        })
    return results
