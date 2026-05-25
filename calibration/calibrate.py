import numpy as np
from scipy.optimize import differential_evolution
from models import HestonParams, MarketData, CalibrationResult
from calibration.objective import heston_rmse

FIXED_KAPPA = 2.0
FIXED_THETA = 0.04

PARAM_BOUNDS = [
    (1e-4, 1.0),   # v0
    (0.01, 2.0),   # sigma
    (-0.99, 0.99), # rho
]
def calibrate(data: MarketData, kappa: float = FIXED_KAPPA, theta: float = FIXED_THETA, n_restarts: int = 3, seed: int = 42) -> CalibrationResult:
    def objective(x):
        params = HestonParams(v0=x[0], kappa=kappa, theta=theta, sigma=x[1], rho=x[2])

        try:
            return heston_rmse(params, data)
        except Exception:
            return 1e6
        
    best_result = None
    for i in range(n_restarts):
        result = differential_evolution(
            objective, 
            bounds = PARAM_BOUNDS,
            seed = seed+i,
            maxiter=500,
            tol = 1e-6,
            polish = True,
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    x = best_result.x
    params = HestonParams(v0=x[0], kappa=kappa, theta=theta, sigma=x[1], rho=x[2])

    return CalibrationResult(
        params=params, 
        rmse=best_result.fun,
        n_iter=best_result.nit,
        success=best_result.success,
        message=best_result.message,
        n_restarts=n_restarts,
    )