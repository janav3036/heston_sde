import numpy as np
import pandas as pd
from models import HestonParams

def _realized_variance(ohlcv: pd.DataFrame) -> pd.Series:
    daily_high = ohlcv["high"].resample("D").max()
    daily_low = ohlcv["low"].resample("D").min()
    log_hl = np.log(daily_high / daily_low)
    rv = (log_hl ** 2) / (4 * np.log(2))
    return rv[rv > 0]


def estimate_from_spot(
    prices_1h: pd.Series,
    ohlcv_1min: pd.DataFrame | None = None,
    snapshot_date: str | None = None,
) -> HestonParams:

    dt = 1 / 252

    if ohlcv_1min is not None:
        rv_daily = _realized_variance(ohlcv_1min)
        daily_closes = ohlcv_1min["close"].resample("D").last().dropna()
        daily_closes = daily_closes[daily_closes > 0]
    else:
        log_ret_1h = np.log(prices_1h / prices_1h.shift(1)).dropna()
        rv_daily = (log_ret_1h ** 2).resample("D").sum()
        rv_daily = rv_daily[rv_daily > 0]
        daily_closes = prices_1h.resample("D").last().dropna()

    rv = rv_daily.values
    if snapshot_date is not None:
        snap_dt = pd.Timestamp(snapshot_date, tz="Asia/Kolkata")
        rv_recent = rv_daily[rv_daily.index <= snap_dt].values
        v0 = float(np.mean(rv_recent[-10:]) / dt) if len(rv_recent) >= 1 else float(np.mean(rv) / dt)
    else:
        v0 = float(np.mean(rv[-10:]) / dt)
    theta = float(np.mean(rv) / dt)

    rv_weekly = rv_daily.resample("W").sum()
    rv_weekly = rv_weekly[rv_weekly > 0].values
    autocorr_w = float(np.corrcoef(rv_weekly[:-1], rv_weekly[1:])[0, 1])
    kappa = float(-np.log(max(autocorr_w, 1e-6)) / (5 * dt))
    kappa = max(kappa, 0.1)

    drv_weekly = np.diff(rv_weekly)
    sigma = float(np.std(drv_weekly) / (np.sqrt(np.mean(rv_weekly)) * 5 * dt))
    sigma = float(np.clip(sigma, 0.01, 5.0))


    daily_ret = np.log(daily_closes / daily_closes.shift(1)).dropna()
    drv_series = rv_daily.diff().dropna()
    aligned = pd.concat([daily_ret.rename("ret"), drv_series.rename("drv")], axis=1).dropna()
    rho = float(np.corrcoef(aligned["ret"], aligned["drv"])[0, 1])
    rho = float(np.clip(rho, -0.99, 0.99))

    return HestonParams(v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho)
