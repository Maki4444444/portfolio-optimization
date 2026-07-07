"""
task3_forecast.py
Task 3: Forecast Future Market Trends using the best-performing model from Task 2.

Unlike Task 2 (which evaluates models on a held-out 2025-2026 test split), this
module refits the chosen model on the FULL historical TSLA series so the forward
forecast uses all available information, then projects `FORECAST_DAYS` trading
days into the future with a 95% confidence interval.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib

# See task1_eda.py for why this is conditional on __main__ rather than
# unconditional at import time -- forcing Agg here would silently break
# inline plotting in any notebook that imports this module.
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_prices, DataFetchError, TICKERS, START_DATE, END_DATE
from task2_models import fit_arima, forecast_arima, make_sequences, build_lstm_model

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

FORECAST_DAYS = 252  # ~12 months of trading days
WINDOW = 60


def load_tsla_close():
    """Load TSLA close series from Task 1's processed CSV, falling back to a
    fresh fetch via data_loader if it isn't present."""
    processed_path = os.path.join(PROCESSED_DIR, "close_prices.csv")
    if os.path.exists(processed_path):
        close = pd.read_csv(processed_path, index_col=0, parse_dates=True)
    else:
        try:
            price_dict = load_prices(TICKERS, START_DATE, END_DATE, use_cache=True)
        except DataFetchError as e:
            print(f"[task3_forecast] Data fetch failed: {e}")
            raise
        close = pd.DataFrame({t: df["Close"] for t, df in price_dict.items()})

    tsla = close["TSLA"].dropna()
    tsla.index = pd.to_datetime(tsla.index)
    return tsla


def forecast_arima_future(tsla, horizon, max_p=5, max_q=5):
    """Refit auto_arima on the full series and forecast forward with a native CI."""
    model = fit_arima(tsla, max_p=max_p, max_q=max_q, trace=False)
    central, conf = forecast_arima(model, horizon)
    return np.asarray(central), conf[:, 0], conf[:, 1], model.order


def random_walk_baseline(tsla, horizon):
    """
    Simple random-walk-with-drift baseline: assumes future daily log returns
    equal the historical average, compounded forward from the last actual
    price. Naive baselines like this are typically MORE reliable than complex
    models at long horizons (even though they're worse at short horizons),
    which is why we use it as a shrinkage anchor for the LSTM forecast below.

    Returns (baseline_price_path, hist_drift, hist_vol) where hist_drift/vol
    are the historical daily log-return mean/std, reused by the LSTM path for
    its own bootstrap noise calibration.
    """
    log_returns = np.diff(np.log(tsla.values))
    hist_drift = float(np.mean(log_returns))
    hist_vol = float(np.std(log_returns))

    last_price = float(tsla.iloc[-1])
    cumulative_drift = hist_drift * np.arange(1, horizon + 1)
    baseline_path = last_price * np.exp(cumulative_drift)
    return baseline_path, hist_drift, hist_vol


def shrinkage_weights(horizon, floor=0.15):
    """Linear decay from 1.0 (day 1) to `floor` (final forecasted day) --
    the weight put on the LSTM forecast vs. the stable baseline as the
    horizon grows."""
    return np.maximum(floor, 1 - (1 - floor) * np.arange(horizon) / max(horizon - 1, 1))


def forecast_lstm_future(tsla, horizon, window=WINDOW, epochs=15, batch_size=64,
                          n_boot=100, seed=42, baseline_floor=0.15):
    """
    Forecast future TSLA prices using an LSTM trained on daily LOG RETURNS
    (mirroring the Task 2 fix -- see task2_models.fit_lstm), then reconstructs
    a price path via cumulative log-return summation, iteratively feeding
    each prediction back in as input for the next step (there's no true
    "previous price" available for genuine future forecasting, unlike Task
    2's walk-forward evaluation).

    Training on returns rather than price levels substantially reduces (but
    does not eliminate) the compounding-drift risk of iterative multi-step
    forecasting, since the model is predicting a stationary, near-zero-mean
    quantity rather than an ever-growing absolute price. As a second line of
    defense, the LSTM's central forecast is additionally "shrunk" toward a
    simple historical-drift random-walk baseline as the horizon grows (see
    `shrinkage_weights`) -- naive baselines are typically MORE reliable than
    complex models at long horizons, even though they're worse at short
    horizons. All n_boot (+1 central) paths are advanced together and
    predicted in a single batched `model.predict()` call per timestep.
    """
    from sklearn.preprocessing import StandardScaler
    from tensorflow import keras

    keras.utils.set_random_seed(seed)

    baseline_path, hist_drift, hist_vol = random_walk_baseline(tsla, horizon)

    log_returns = np.diff(np.log(tsla.values))
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(log_returns.reshape(-1, 1))

    X, y = make_sequences(scaled_returns, window)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = build_lstm_model(window=window)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=0.1)

    last_window = scaled_returns[-window:].flatten()

    rng = np.random.default_rng(seed)
    n_paths = n_boot + 1  # path 0 is the noiseless central LSTM path

    windows = np.tile(last_window, (n_paths, 1))
    ret_paths_real = np.zeros((n_paths, horizon))

    # per-step noise in REAL log-return space, calibrated to historical daily
    # volatility -- summed over `horizon` steps this naturally produces the
    # textbook sqrt(time) confidence-interval growth of a random walk
    noise_scale = hist_vol * 0.5

    for step in range(horizon):
        batch_x = windows.reshape(n_paths, window, 1)
        preds_scaled = model.predict(batch_x, verbose=0).flatten()
        preds_real = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

        noise = rng.normal(0, noise_scale, size=n_paths)
        noise[0] = 0.0  # keep path 0 as the noiseless central path
        preds_real_noisy = preds_real + noise

        ret_paths_real[:, step] = preds_real_noisy

        preds_scaled_noisy = scaler.transform(preds_real_noisy.reshape(-1, 1)).flatten()
        windows = np.concatenate([windows[:, 1:], preds_scaled_noisy.reshape(-1, 1)], axis=1)

    last_price = float(tsla.iloc[-1])
    cumulative_log_returns = np.cumsum(ret_paths_real, axis=1)
    lstm_price_paths = last_price * np.exp(cumulative_log_returns)  # shape (n_paths, horizon)

    # shrink every path toward the stable baseline as the horizon grows
    w = shrinkage_weights(horizon, floor=baseline_floor)  # shape (horizon,)
    blended_paths = w * lstm_price_paths + (1 - w) * baseline_path

    central = blended_paths[0]
    boot_paths = blended_paths[1:]
    lower = np.percentile(boot_paths, 2.5, axis=0)
    upper = np.percentile(boot_paths, 97.5, axis=0)

    return central, lower, upper


def analyze_forecast(tsla, forecast_df, model_label):
    """Compute trend direction, CI-width growth, and simple opportunity/risk flags."""
    start_price = float(tsla.iloc[-1])
    end_price = float(forecast_df["forecast"].iloc[-1])
    total_return_pct = (end_price / start_price - 1) * 100

    ci_width = forecast_df["upper_95"] - forecast_df["lower_95"]
    ci_width_start = float(ci_width.iloc[0])
    ci_width_end = float(ci_width.iloc[-1])
    ci_growth_factor = ci_width_end / ci_width_start if ci_width_start > 0 else float("nan")

    max_drawup_pct = float((forecast_df["forecast"].max() / start_price - 1) * 100)
    max_drawdown_pct = float((forecast_df["forecast"].min() / start_price - 1) * 100)

    if total_return_pct > 5:
        trend_direction = "upward"
    elif total_return_pct < -5:
        trend_direction = "downward"
    else:
        trend_direction = "roughly flat"

    return {
        "model_used": model_label,
        "horizon_trading_days": len(forecast_df),
        "last_actual_price": start_price,
        "forecast_end_price": end_price,
        "forecast_total_return_pct": total_return_pct,
        "trend_direction": trend_direction,
        "max_forecast_upside_pct": max_drawup_pct,
        "max_forecast_downside_pct": max_drawdown_pct,
        "ci_width_day_1": ci_width_start,
        "ci_width_day_final": ci_width_end,
        "ci_growth_factor": ci_growth_factor,
    }


def plot_forecast(tsla, forecast_df, model_label, out_path=None):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    hist = tsla[-252:]
    ax.plot(hist.index, hist.values, label="Historical (last 12mo)", color="black")
    ax.plot(forecast_df.index, forecast_df["forecast"], label=f"{model_label} Forecast", color="tab:orange")
    ax.fill_between(forecast_df.index, forecast_df["lower_95"], forecast_df["upper_95"],
                     alpha=0.2, color="tab:orange", label="95% CI")
    ax.axvline(tsla.index.max(), color="gray", linestyle=":", linewidth=1)
    ax.set_title(f"TSLA {len(forecast_df)}-Trading-Day Forecast ({model_label})")
    ax.set_xlabel("Date"); ax.set_ylabel("Price ($)")
    ax.legend(); fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
    return fig


def run(model_choice="LSTM"):
    tsla = load_tsla_close()
    future_dates = pd.bdate_range(start=tsla.index.max() + pd.Timedelta(days=1), periods=FORECAST_DAYS)

    if model_choice.upper() == "ARIMA":
        central, lower, upper, order = forecast_arima_future(tsla, FORECAST_DAYS)
        label = f"ARIMA{order}"
    else:
        central, lower, upper = forecast_lstm_future(tsla, FORECAST_DAYS)
        label = "LSTM"

    forecast_df = pd.DataFrame(
        {"forecast": central, "lower_95": lower, "upper_95": upper}, index=future_dates
    )
    forecast_df.to_csv(os.path.join(PROCESSED_DIR, "task3_future_forecast.csv"))

    analysis = analyze_forecast(tsla, forecast_df, label)
    with open(os.path.join(PROCESSED_DIR, "task3_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)

    plot_forecast(tsla, forecast_df, label, out_path=os.path.join(FIG_DIR, "06_future_forecast.png"))

    print(json.dumps(analysis, indent=2))
    return analysis


if __name__ == "__main__":
    run()