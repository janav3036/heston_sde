import numpy as np
import pandas as pd
from pathlib import Path
from models import MarketData

OPTIONS_DIR = Path(__file__).parent.parent / "options"
DEFAULT_R = 0.065  # ~91-day India T-bill rate
DEFAULT_Q = 0.0

_OTM_FILES = [
    "ATM-3_PUT.parquet",
    "ATM-2_PUT.parquet",
    "ATM-1_PUT.parquet",
    "ATM_CALL.parquet",
    "ATM+1_CALL.parquet",
    "ATM+2_CALL.parquet",
    "ATM+3_CALL.parquet",
]

def _next_thursday(dt: pd.Timestamp) -> pd.Timestamp:
    days = (3 - dt.weekday()) % 7 or 7
    return (dt + pd.Timedelta(days=days)).normalize()

def load_snapshot(
    snapshot_dt: str,
    r: float = DEFAULT_R,
    q: float = DEFAULT_Q,
) -> MarketData:
    dt = pd.Timestamp(snapshot_dt, tz="Asia/Kolkata")

    strikes, ivs, spots = [], [], []
    for fname in _OTM_FILES:
        df = pd.read_parquet(OPTIONS_DIR / fname)
        idx = df.index.get_indexer([dt], method="nearest")[0]
        row = df.iloc[idx]
        strikes.append(float(row["strike_val"]))
        ivs.append(float(row["iv"]) / 100.0)
        spots.append(float(row["spot"]))
    S = float(np.median(spots))
    T = max((_next_thursday(dt) - dt).days / 365.25, 1 / 365)

    order = np.argsort(strikes)
    strikes = np.array(strikes)[order]
    ivs = np.array(ivs)[order]

    strikes_rev = strikes[::-1]
    ivs_rev = ivs[::-1]
    _, unique_idx = np.unique(strikes_rev, return_index=True)
    strikes = strikes_rev[unique_idx]
    ivs = ivs_rev[unique_idx]



    return MarketData(
        S=S,
        r=r,
        q=q,
        strikes=strikes,
        expiries=np.array([T]),
        market_ivs=ivs[np.newaxis, :],
    )

def load_spot(ticker: str) -> pd.Series:
    path = Path(__file__).parent.parent / "1h" / f"{ticker}.parquet"
    df = pd.read_parquet(path)
    return df["Close"].sort_index().dropna()


_TICKER_1MIN = {
    "NSEI":         "nifty_spot_1min.parquet",
    "RELIANCE_NS":  "RELIANCE_spot_1min.parquet",
    "HDFCBANK_NS":  "HDFCBANK_spot_1min.parquet",
    "ICICIBANK_NS": "ICICIBANK_spot_1min.parquet",
    "INFY_NS":      "INFY_spot_1min.parquet",
    "TCS_NS":       "TCS_spot_1min.parquet",
}

def load_spot_1min(ticker: str) -> pd.Series:
    fname = _TICKER_1MIN[ticker]
    path = Path(__file__).parent.parent / "spot_1min" / fname
    df = pd.read_parquet(path)
    return df["close"].sort_index().dropna()


def load_ohlcv_1min(ticker: str) -> pd.DataFrame:
    fname = _TICKER_1MIN[ticker]
    path = Path(__file__).parent.parent / "spot_1min" / fname
    df = pd.read_parquet(path)
    return df[["open", "high", "low", "close"]].sort_index().dropna()
