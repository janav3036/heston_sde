import numpy as np
import pytest
from models.black_scholes import bs_call, bs_put, bs_vega, implied_vol

S, r, q, T = 100.0, 0.05, 0.0, 1.0
K = np.array([90.0, 100.0, 110.0])
SIGMA = np.array([0.2, 0.2, 0.2])


def test_bs_call_put_parity():
    call = bs_call(S, K, r, q, T, SIGMA)
    put = bs_put(S, K, r, q, T, SIGMA)
    parity = call - put - (S * np.exp(-q * T) - K * np.exp(-r * T))
    np.testing.assert_allclose(parity, 0.0, atol=1e-10)


def test_bs_call_intrinsic_bound():
    call = bs_call(S, K, r, q, T, SIGMA)
    assert np.all(call >= np.maximum(S * np.exp(-q * T) - K * np.exp(-r * T), 0))


def test_bs_vega_positive():
    vega = bs_vega(S, K, r, q, T, SIGMA)
    assert np.all(vega > 0)


def test_implied_vol_roundtrip():
    prices = bs_call(S, K, r, q, T, SIGMA)
    iv = implied_vol(prices, S, K, r, q, T)
    np.testing.assert_allclose(iv, SIGMA, atol=1e-6)


def test_implied_vol_put_roundtrip():
    from models.black_scholes import bs_put
    prices = bs_put(S, K, r, q, T, SIGMA)
    iv = implied_vol(prices, S, K, r, q, T, option_type="put")
    np.testing.assert_allclose(iv, SIGMA, atol=1e-6)
