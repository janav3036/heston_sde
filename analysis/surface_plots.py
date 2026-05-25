import numpy as np
import matplotlib.pyplot as plt
from models import HestonParams, MarketData
from models.heston_fft import carr_madan_price, price_at_strikes
from models.black_scholes import implied_vol

def plot_smile(
        params: HestonParams,
        data: MarketData,
        expiry_idx: int = 0,
        ax: plt.Axes = None,
    ) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8,5))

    T = data.expiries[expiry_idx]
    market_ivs = data.market_ivs[expiry_idx]

    strikes_fft, prices_fft = carr_madan_price(params, data.S, data.r, data.q, T)
    model_prices = price_at_strikes(strikes_fft, prices_fft, data.strikes)
    model_ivs = implied_vol(model_prices, data.S, data.strikes, data.r, data.q, T)

    moneyness = data.strikes / data.S

    ax.plot(moneyness, market_ivs * 100, "o", label="Market", color="steelblue")
    ax.plot(moneyness, model_ivs * 100, "-", label="Heston", color="tomato")
    ax.set_xlabel("Moneyness (K/S)")
    ax.set_ylabel("Implied Volatility (%)")
    ax.set_title(f"Vol Smile — T = {T:.4f}y")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_surface(
    params: HestonParams,
    data: MarketData,
    ) -> plt.Figure:
    fig, axes = plt.subplots(
        1, len(data.expiries),
        figsize=(6 * len(data.expiries), 5),
        squeeze=False,
    )
    for i in range(len(data.expiries)):
        plot_smile(params, data, expiry_idx=i, ax=axes[0, i])

    fig.tight_layout()
    return fig
