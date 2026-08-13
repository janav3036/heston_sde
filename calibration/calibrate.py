import numpy as np
from scipy.optimize import differential_evolution
from models import HestonParams, MarketData, CalibrationResult
from calibration.objective import heston_rmse

PARAM_BOUNDS = [
    (1e-4, 1.0),    # v0
    (0.5,  5.0),   # kappa
    (1e-4, 1.0),    # theta
    (0.01, 5.0),    # sigma  (widened from 2.0)
    (-0.99, 0.99),  # rho
]

def calibrate(data: MarketData, n_restarts: int = 3, seed: int = 42) -> CalibrationResult:
    def objective(x):
        params = HestonParams(v0=x[0], kappa=x[1], theta=x[2], sigma=x[3], rho=x[4])

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
    params = HestonParams(v0=x[0], kappa=x[1], theta=x[2], sigma=x[3], rho=x[4])

    return CalibrationResult(
        params=params, 
        rmse=best_result.fun,
        n_iter=best_result.nit,
        success=best_result.success,
        message=best_result.message,
        n_restarts=n_restarts,
    )