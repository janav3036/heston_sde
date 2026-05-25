from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class HestonParams:
    """
    Five parameters of the Heston Stochastic Volatility model

    dS = mu*S dt + sqrt(v)*S dw^S
    dv = kappa*(theta-v) dt + sigma*sqrt(v) dW^v
    corr(dW^S, dW^v) = rho dt

    Feller condition (Variance stays +ve) -> 2*kappa*theta > sigma**2

    """
    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float

    def feller_satisfied(self) -> bool:
        return 2*self.kappa*self.theta > self.sigma**2
    
    def feller_violation(self) -> float: 
        """Positive value means condition is violated"""
        return self.sigma**2-2*self.kappa*self.theta
    
@dataclass
class MarketData:
    """
    Market Snapshot for one calibration run

    market_ivs has shape(len(expiries), len(strikes))
    NaN marks illiquid strikes that should be excluded from calibration
    """
    S: float
    r: float
    q: float
    strikes: np.ndarray
    expiries: np.ndarray
    market_ivs: np.ndarray
    snapshot_date: Optional[str] = field(default=None)
    source: Optional[str] = field(default="NSE")

@dataclass
class CalibrationResult:
    """
    Output of a single calibration
    """

    params: HestonParams
    rmse: float
    n_iter: int
    success: bool
    message: str
    n_restarts: int