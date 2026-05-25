import numpy as np
import pytest
from models import HestonParams
from models.heston_fft import carr_madan_price, price_at_strikes

S, r, q, T = 100.0, 0.05, 0.0, 1.0
PARAMS = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=0.7)

def test_prices_non_negative():
    strikes, prices = carr_madan_price(PARAMS, S, r, q, T)
    assert np.all(prices>=0)

def test_atm_price_positive():
    strikes, prices = carr_madan_price(PARAMS, S, r, q, T)
    atm_idx = np.argmin(np.abs(strikes - 1.0)) # since strikes are moneyness K/S on log scale
    assert prices[atm_idx] > 0

def test_price_at_strikes_interpolation():
    strikes_fft, prices_fft = carr_madan_price(PARAMS, S, r, q, T)
    target = np.array([0.9, 1.0, 1.1]) * S
    interp = price_at_strikes(strikes_fft * S, prices_fft, target)
    assert interp.shape == (3,)
    assert np.all(interp >= 0)