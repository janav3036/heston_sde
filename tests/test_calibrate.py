import numpy as np
import pandas as pd
import pytest
from models import HestonParams, MarketData
from models.heston_fft import carr_madan_price, price_at_strikes
from models.black_scholes import implied_vol
from calibration.calibrate import calibrate
from calibration.stability import estimate_from_spot

S, r, q, T = 100.0, 0.05, 0.0, 1.0
TRUE_PARAMS = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)

def _make_synthetic_data(params: HestonParams) -> MarketData:
    strikes_fft, prices_fft = carr_madan_price(params, S, r, q, T)
    target_strikes = np.array([0.9, 0.95, 1.0, 1.05, 1.10]) * S
    prices = price_at_strikes(strikes_fft * S, prices_fft, target_strikes)
    ivs = implied_vol(prices, S, target_strikes, r, q, T)
    return MarketData(
        S=S, 
        r=r,
        q=q,
        strikes=target_strikes,
        expiries=np.array([T]),
        market_ivs=ivs[np.newaxis, :]
        
    )

def test_calibrate_low_rmse():
    data = _make_synthetic_data(TRUE_PARAMS)
    result = calibrate(data, kappa=TRUE_PARAMS.kappa, theta=TRUE_PARAMS.theta)
    assert result.rmse < 0.005

def test_calibrate_result_fields():
    data = _make_synthetic_data(TRUE_PARAMS)
    result = calibrate(data, kappa=TRUE_PARAMS.kappa, theta=TRUE_PARAMS.theta)
    assert hasattr(result, "params")
    assert hasattr(result, "rmse")
    assert hasattr(result, "success")
    assert 0 < result.params.v0 < 1
    assert -0.99 < result.params.rho < 0.99

def test_estimate_from_spot_reasonable():
    rng = np.random.default_rng(42)
    n = 500
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="Asia/Kolkata")
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    params = estimate_from_spot(prices)
    assert 0 < params.v0
    assert 0 < params.kappa
    assert 0 < params.theta
    assert 0 < params.sigma
    assert -1 < params.rho < 1


