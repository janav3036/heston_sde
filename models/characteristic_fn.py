import numpy as np
from models import HestonParams

IMAGINARY_UNIT = 1j

def heston_char_fn(
        omega: np.ndarray,
        params: HestonParams,
        S: float,
        r:float,
        q:float,
        T: float
    ) -> np.ndarray:
    """
    Heston 1993 characteristic function of log(S_T), using the Albrecher
    'little trap' formation to avoid branch cut discontinuities in the complex logarithm
    
    Parameters:
        omega: array of frequencies
        params: HestonParams
        S: spot price
        r: risk free rate
        q: dividend yield
        T: time to expiry in years
    
    Returns:
        Complpex array, same shape as omega
        phi(0) = 1 because of normalisation property of characteristic function
    """

    v0 = params.v0
    kappa = params.kappa
    theta = params.theta
    sigma = params.sigma
    rho = params.rho

    x0 = np.log(S) + (r - q) * T
    omega = np.asarray(omega, dtype=complex)
    zero_mask = (omega.real == 0) & (omega.imag == 0)
    omega_safe = np.where(zero_mask, 1.0, omega)

    b = kappa - rho * sigma * IMAGINARY_UNIT * omega_safe
    d = np.sqrt(b**2 + sigma**2 * omega_safe * (omega_safe + IMAGINARY_UNIT))

    g2 = (b + d) / (b - d)
    exp_dT = np.exp(-d * T)
    log_term = np.log((g2 - exp_dT) / (g2 - 1))

    phi = np.exp(
        IMAGINARY_UNIT * omega_safe * x0
        + (kappa * theta / sigma**2) * ((b - d) * T - 2 * log_term)
        + (v0 / sigma**2) * (b + d) * (1 - exp_dT) / (g2 - exp_dT)
    )

    phi = np.where(zero_mask, 1.0 + 0j, phi)
    return phi
