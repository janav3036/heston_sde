import numpy as np
from models import HestonParams, MarketData
from models.heston_fft import carr_madan_price, price_at_strikes
from models.black_scholes import implied_vol

FELLER_PENALTY = 0.1

def heston_rmse(params: HestonParams, data: MarketData) -> float:
    errors = []
    for i, T in enumerate(data.expiries):
        strikes_fft, prices_fft = carr_madan_price(params, data.S, data.r, data.q, T)
        model_prices = price_at_strikes(strikes_fft * data.S, prices_fft, data.strikes)
        model_ivs = implied_vol(model_prices, data.S, data.strikes, data.r, data.q, T)

        market_ivs = data.market_ivs[i]
        mask = ~np.isnan(market_ivs) & ~np.isnan(model_ivs)
        diff = model_ivs[mask] - market_ivs[mask]
        errors.append(diff)

    all_errors = np.concatenate(errors)
    rmse = float(np.sqrt(np.mean(all_errors**2)))

    feller_violation = max(0.0, params.sigma**2 - 2 * params.kappa * params.theta)
    return rmse + FELLER_PENALTY * feller_violation
