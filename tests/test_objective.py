import numpy as np
import pytest

from models import HestonParams, MarketData
from models.heston_fft import carr_madan_price, price_at_strikes
from models.black_scholes import implied_vol 
from calibration.objective import heston_rmse

S, r, q, T = 100.0, 0.05, 0.0, 1.0
TRUE_PARAMS = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)

def _make_synthetic_data(params: HestonParams) -> MarketData:
    strikes_fft, prices_fft = carr_madan_price(params, S, r, q, T)
    target_strikes = np.array([0.9, 0.95, 1.0, 1.05, 1.10]) * S
    prices = price_at_strikes(strikes_fft * S, prices_fft, target_strikes)
    ivs = implied_vol(prices, S, target_strikes, r, q, T)
    return MarketData(
        S=S, r=r, q=q,
        strikes = target_strikes,
        expiries = np.array([T]),
        market_ivs = ivs[np.newaxis, :]
    )

def test_rmse_near_zero_for_true_params():
    data = _make_synthetic_data(TRUE_PARAMS)
    rmse = heston_rmse(TRUE_PARAMS, data)
    assert rmse < 1e-4

def test_wrong_params_give_higher_rmse():
    data = _make_synthetic_data(TRUE_PARAMS)
    wrong = HestonParams(v0=0.09, kappa=1.0, theta=0.09, sigma=0.6, rho=0.0)
    assert heston_rmse(wrong, data) > heston_rmse(TRUE_PARAMS, data)