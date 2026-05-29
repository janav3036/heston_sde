import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data.nse_loader import load_snapshot, load_ohlcv_1min, load_spot
from calibration.stability import estimate_from_spot
from calibration.calibrate import calibrate
from calibration.objective import heston_rmse 
from analysis.surface_plots import plot_smile
from analysis.error_analysis import summary_table, bias_by_moneyness 

SNAPSHOT = "2024-11-14 10:00:00"
TICKER = "NSEI"

def main():
    print(f"Loading snapshot: {SNAPSHOT}")
    data = load_snapshot(SNAPSHOT)
    print(f" S={data.S:.1f}, T={data.expiries[0]:.4f}y, strikes={data.strikes}")

    print("Estimating intial params from spot data...")
    prices_1h = load_spot(TICKER)
    ohlcv = load_ohlcv_1min(TICKER)
    init = estimate_from_spot(prices_1h, ohlcv, snapshot_date=SNAPSHOT)
    print(f"  Init: {init}")
    
    print("Calibrating...")
    result = calibrate(data, kappa=init.kappa, theta=init.theta)
    print(f"    RMSE : {result.rmse:.6f}")
    print(f"    Params : {result.params}")
    print(f"    Feller satisfied: {result.params.feller_satisfied()}")

    print("\nSummary Table:")
    print(summary_table(result.params, data).to_string(index=False))

    print("\nBias by Moneyness:")
    print(bias_by_moneyness(result.params, data).to_string(index=False) )

    fig, ax = plt.subplots(figsize=(8,5))
    plot_smile(result.params, data, ax=ax)
    fig.savefig("smile.png", dpi=150, bbox_inches="tight")
    print("\nSmile plot saved to smile.png")

if __name__ == "__main__":
    main()