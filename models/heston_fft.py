import numpy as np
from scipy.interpolate import CubicSpline
from models import HestonParams
from models.characteristic_fn import heston_char_fn

#FFT grid parameters 
N_FFT = 4096 # grid size, must be power of 2 for efficiency
ETA = 0.25 # frequency domain integration step
ALPHA = 1.5 # dampening parameter, stable range = [1.0,2.0]
LAMBDA = 2 * np.pi / (N_FFT*ETA) #log strike spacing - Nyquist relation
B = N_FFT*LAMBDA/2 #log strike range = [-B, B]

def carr_madan_price(
        params: HestonParams,
        S: float,
        r: float,
        q: float,
        T: float,
        alpha: float = ALPHA,
        n_fft: int = N_FFT,
        eta: float = ETA,
    ) -> tuple:
    """
    Pricing European calls via Carr-Madan FFT pricing
    
    C(k) = (e^{-alpha*k} / pi) * Re[ FFT(psi) ](k)

    where k = ln(K/S) is log-moneyness and:
        psi(omega) = e^{-rT} * phi(omega - (alpha+1)*i)
                     / (alpha^2 + alpha - omega^2 + i*(2*alpha+1)*omega)

    Returns: 
        strikes: np.ndarray
            Strike prices on the FFT grid
        prices: np.ndarray
            Call prices corresponding to the strike, clipped @ 0
    """

    lam = 2*np.pi / (n_fft*eta)
    b = n_fft*lam/2

    omega = np.arange(n_fft)*eta #frequency grid

    j = np.arange(n_fft) # simspson's rule weights for integration accuracy
    w = (eta/3)*(3+(-1)**j)
    w[0] -= eta/3

    psi = np.zeros(n_fft, dtype=complex) # dampened integrand
    shifted_omega = omega - (alpha+1) * 1j
    phi = heston_char_fn(shifted_omega, params, S, r, q, T)
    denom = alpha**2 + alpha - omega**2 +1j*(2*alpha +1)*omega
    psi = w * np.exp(-r * T) * phi / denom * np.exp(1j * omega * b)


    #FFT and recover call prices
    fft_vals = np.fft.fft(psi)
    log_strikes = -b + lam * np.arange(n_fft)
    k_grid = log_strikes
    multiplier = np.exp(-alpha * k_grid)/np.pi
    prices = multiplier*np.real(fft_vals)
    prices = np.clip(prices, 0, None)

    strikes = np.exp(k_grid)
    return strikes, prices



def price_at_strikes(
        strikes_fft: np.ndarray,
        prices_fft: np.ndarray,
        target_strikes: np.ndarray,
    ) -> np.ndarray:
    """
    Interpolate from the FFT strike grid to specific market strikes.

    Uses cubic spline (C2 smooth) rather than linear interpolation —
    linear interpolation of option prices introduces systematic bias
    when inverting to implied volatilities.

    Parameters
    ----------
    strikes_fft    : strike grid from carr_madan_price
    prices_fft     : call prices from carr_madan_price
    target_strikes : market strikes to evaluate at

    Returns
    -------
    Call prices at target_strikes.
    """
    cs = CubicSpline(strikes_fft, prices_fft)
    return cs(target_strikes)
