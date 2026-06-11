from dataclasses import dataclass

@dataclass
class OptionParams:
    S: float      # spot price
    K: float      # strike price
    T: float      # time to expiry (years)
    r: float      # risk-free rate
    sigma: float  # volatility
    q: float = 0.0  # continuous dividend yield
