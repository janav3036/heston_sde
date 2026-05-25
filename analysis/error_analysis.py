import numpy as np
import pandas as pd
from models import HestonParams, MarketData
from models.heston_fft import carr_madan_price, price_at_strikes
from models.black_scholes import implied_vol


def _model_ivs(params: HestonParams, data: MarketData, expiry_idx: int) -> np.ndarray:
    T = data.expiries[expiry_idx]
    strikes_fft, prices_fft = carr_madan_price(params, data.S, data.r, data.q, T)
    model_prices = price_at_strikes(strikes_fft, prices_fft, data.strikes)
    return implied_vol(model_prices, data.S, data.strikes, data.r, data.q, T)


def expiry_errors(params: HestonParams, data: MarketData) -> list[pd.DataFrame]:
    """Per-strike breakdown for every expiry. Each DataFrame has columns:
    strike, moneyness, market_iv, model_iv, error, abs_error.
    Rows where market IV is NaN are dropped."""
    results = []
    for i, T in enumerate(data.expiries):
        market_ivs = data.market_ivs[i]
        model_ivs = _model_ivs(params, data, i)
        mask = ~np.isnan(market_ivs) & ~np.isnan(model_ivs)
        df = pd.DataFrame({
            "strike":     data.strikes[mask],
            "moneyness":  data.strikes[mask] / data.S,
            "market_iv":  market_ivs[mask],
            "model_iv":   model_ivs[mask],
            "error":      model_ivs[mask] - market_ivs[mask],
            "abs_error":  np.abs(model_ivs[mask] - market_ivs[mask]),
        })
        df.attrs["T"] = T
        results.append(df)
    return results


def summary_table(params: HestonParams, data: MarketData) -> pd.DataFrame:
    """One row per expiry: expiry (years), RMSE, mean_error (signed bias), max_abs_error."""
    rows = []
    for df in expiry_errors(params, data):
        rows.append({
            "expiry_y":     df.attrs["T"],
            "n_strikes":    len(df),
            "rmse":         float(np.sqrt(np.mean(df["error"] ** 2))),
            "mean_error":   float(df["error"].mean()),
            "max_abs_error": float(df["abs_error"].max()),
        })
    return pd.DataFrame(rows)


def bias_by_moneyness(
    params: HestonParams,
    data: MarketData,
    otm_put_cutoff: float = 0.97,
    otm_call_cutoff: float = 1.03,
) -> pd.DataFrame:
    """Average signed error split into OTM put / ATM / OTM call buckets, per expiry.
    Buckets: moneyness < otm_put_cutoff, between cutoffs (ATM), > otm_call_cutoff."""
    records = []
    for df in expiry_errors(params, data):
        put_mask  = df["moneyness"] < otm_put_cutoff
        call_mask = df["moneyness"] > otm_call_cutoff
        atm_mask  = ~put_mask & ~call_mask
        records.append({
            "expiry_y":      df.attrs["T"],
            "otm_put_bias":  float(df.loc[put_mask,  "error"].mean()) if put_mask.any()  else float("nan"),
            "atm_bias":      float(df.loc[atm_mask,  "error"].mean()) if atm_mask.any()  else float("nan"),
            "otm_call_bias": float(df.loc[call_mask, "error"].mean()) if call_mask.any() else float("nan"),
        })
    return pd.DataFrame(records)
