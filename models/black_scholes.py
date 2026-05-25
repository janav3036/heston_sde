import numpy as np
from scipy.stats import norm

VEGA_TOL = 1e-10

def bs_call(
        S: float,
        K: np.ndarray,
        r: float,
        q: float,
        T: float,
        sigma: np.ndarray,
) -> np.ndarray:
    """
    Black-Scholes European call price (Garman-Kohlhagen form, includes dividend yield q).

    C = S*e^{-qT}*N(d1) - K*e^{-rT}*N(d2)
    d1 = (ln(S/K) + (r - q + 0.5*sigma^2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    """

    d1 = (np.log(S/K) + (r-q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    return S * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put(
    S: float,
    K: np.ndarray,
    r: float,
    q: float,
    T: float,
    sigma: np.ndarray,
) -> np.ndarray:
    """
    Black-Scholes European put price.

    P = K*e^{-rT}*N(-d2) - S*e^{-qT}*N(-d1)
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def bs_vega(
    S: float,
    K: np.ndarray,
    r: float,
    q: float,
    T: float,
    sigma: np.ndarray,
) -> np.ndarray:
    """
    Vega = dC/d(sigma) = S * e^{-qT} * N'(d1) * sqrt(T)

    N'(x) is the standard normal PDF.
    Same formula for calls and puts — vega is identical for both.
    Used as the denominator in Newton-Raphson IV inversion.
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

def implied_vol(
    price: np.ndarray,
    S: float,
    K: np.ndarray,
    r: float,
    q: float,
    T: float,
    option_type: str = "call",
    max_iter: int = 100,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Newton-Raphson implied volatility inversion.

    sigma_{n+1} = sigma_n - (BS(sigma_n) - price) / vega(sigma_n)

    Returns NaN where inversion fails (vega too small — deep ITM/OTM).
    Vectorised: all strikes are updated simultaneously each iteration.
    """
    # Convert puts to calls via put-call parity
    if option_type == "put":
        price = price + S * np.exp(-q * T) - K * np.exp(-r * T)

    # Brenner-Subrahmanyam initial guess — close to true IV for ATM options
    sigma = np.sqrt(2 * np.pi / T) * price / S
    sigma = np.where(sigma > 0, sigma, 0.2 * np.ones_like(sigma))

    for _ in range(max_iter):
        vega = bs_vega(S, K, r, q, T, sigma)
        price_diff = bs_call(S, K, r, q, T, sigma) - price

        # Stop updating where vega is too small — would cause division blowup
        valid = vega > VEGA_TOL
        update = np.where(valid, price_diff / vega, 0.0)
        sigma = sigma - update

        if np.max(np.abs(price_diff[valid])) < tol:
            break

    # Mark failed inversions as NaN
    sigma = np.where(valid, sigma, np.nan)
    return sigma