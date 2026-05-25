import numpy as np
import pytest
from models import HestonParams
from models.characteristic_fn import heston_char_fn

S, r, q, T = 100.0, 0.05, 0.0, 1.0
PARAMS = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)

def test_phi_at_zero_is_one():
    omega = np.array([0.0])
    phi = heston_char_fn(omega, PARAMS, S, r, q, T)
    np.testing.assert_allclose(np.abs(phi), 1.0, atol = 1e-10)

def test_phi_returns_complex():
    omega = np.linspace(0.1, 10.0, 50)
    phi = heston_char_fn(omega, PARAMS, S, r, q, T)
    assert phi.dtype == complex or np.iscomplexobj(phi)

def test_phi_modulus_bounded():
    # |phi(omega)| <= 1 for all real omega
    omega = np.linspace(0, 50, 200)
    phi = heston_char_fn(omega, PARAMS, S, r, q, T)
    assert np.all(np.abs(phi) <= 1.0 + 1e-9)