"""
task2_models.py
Task 2: Build Time Series Forecasting Models (ARIMA + LSTM) for TSLA close price.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib

# See task1_eda.py for why this is conditional on __main__ rather than
# unconditional at import time -- forcing Agg here would silently break
# inline plotting in any notebook that imports this module.
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(FIG_DIR, exist_ok=True)

TRAIN_END = "2024-12-31"  # train: 2015-2024, test: 2025-2026 (per brief)
WINDOW = 60


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def load_close():
    close = pd.read_csv(os.path.join(PROCESSED_DIR, "close_prices.csv"), index_col=0, parse_dates=True)
    return close["TSLA"]


def chrono_split(series, train_end=TRAIN_END):
    train = series[series.index <= train_end]
    test = series[series.index > train_end]
    return train, test


# ---------------- ARIMA ----------------

def fit_arima(train, max_p=3, max_q=3, trace=False):
    model = pm.auto_arima(
        train, start_p=0, start_q=0, max_p=max_p, max_q=max_q, d=None,
        seasonal=False, stepwise=True, suppress_warnings=True,
        error_action="ignore", trace=trace, n_jobs=1, maxiter=50,
    )
    return model


def forecast_arima(model, n_periods):
    fc, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)
    return np.asarray(fc), conf_int


# ---------------- LSTM ----------------

def make_sequences(values, window=WINDOW):
    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i - window:i, 0])
        y.append(values[i, 0])
    return np.array(X), np.array(y)


def build_lstm_model(window=WINDOW):
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(window, 1)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def fit_lstm(train, test, window=WINDOW, epochs=12, batch_size=64, return_history=False):
    """
    Trains the LSTM to predict next-day LOG RETURNS rather than raw price
    levels. This mirrors the same reasoning that requires ARIMA to difference
    the series (Task 1's ADF tests: price is non-stationary, returns are
    stationary) -- a price-level LSTM has the same underlying mismatch, it's
    just less visible here because test predictions use the TRUE historical
    price immediately preceding each test day as input (a walk-forward
    one-step evaluation, not recursive multi-step feedback).

    Test predictions are converted back to price level via
    next_price = true_previous_price * exp(predicted_log_return), so the
    returned `preds` remain directly comparable (in price units) to ARIMA's
    forecast and the actual test prices.
    """
    from sklearn.preprocessing import StandardScaler

    full_prices = pd.concat([train, test])
    log_prices = np.log(full_prices.values)
    log_returns = np.diff(log_prices)  # log_returns[i] = return from full_prices[i] to full_prices[i+1]

    n_train = len(train)
    n_train_returns = n_train - 1  # returns fully contained within the train period

    scaler = StandardScaler()
    train_returns_scaled = scaler.fit_transform(log_returns[:n_train_returns].reshape(-1, 1))
    all_returns_scaled = scaler.transform(log_returns.reshape(-1, 1))

    X_train, y_train = make_sequences(train_returns_scaled, window)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    # test sequences: windows of TRUE historical returns ending right before each test day
    test_input = all_returns_scaled[n_train_returns - window:]
    X_test, y_test = make_sequences(test_input, window)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = build_lstm_model(window)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0,
                         validation_split=0.1)

    pred_returns_scaled = model.predict(X_test, verbose=0)
    pred_returns = scaler.inverse_transform(pred_returns_scaled).flatten()

    # reconstruct price predictions using the TRUE previous close (walk-forward,
    # not recursively-generated) -- no compounding drift at this evaluation stage
    prev_prices = full_prices.values[n_train_returns:n_train_returns + len(pred_returns)]
    preds = prev_prices * np.exp(pred_returns)

    if return_history:
        return model, scaler, preds, history
    return model, scaler, preds


def run():
    close = load_close()
    train, test = chrono_split(close)
    print(f"Train: {train.index.min().date()} -> {train.index.max().date()} ({len(train)} obs)")
    print(f"Test:  {test.index.min().date()} -> {test.index.max().date()} ({len(test)} obs)")

    # ARIMA
    arima_model = fit_arima(train)
    arima_fc, arima_conf = forecast_arima(arima_model, len(test))
    arima_order = arima_model.order

    # LSTM
    lstm_model, scaler, lstm_preds = fit_lstm(train, test)

    results = {}
    for name, preds in [("ARIMA", arima_fc), ("LSTM", lstm_preds)]:
        y_true = test.values[:len(preds)]
        y_pred = preds[:len(y_true)]
        results[name] = {
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAPE": mape(y_true, y_pred),
        }
    results["ARIMA"]["order"] = list(arima_order)

    best_model = min(results, key=lambda k: results[k]["RMSE"])
    results["best_model"] = best_model

    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train.index[-200:], train.values[-200:], label="Train (last 200d)", color="gray")
    ax.plot(test.index, test.values, label="Actual (test)", color="black", linewidth=1.2)
    ax.plot(test.index[:len(arima_fc)], arima_fc, label=f"ARIMA{arima_order}", linestyle="--")
    ax.fill_between(test.index[:len(arima_fc)], arima_conf[:, 0], arima_conf[:, 1], alpha=0.15, label="ARIMA 95% CI")
    ax.plot(test.index[:len(lstm_preds)], lstm_preds, label="LSTM", linestyle="--")
    ax.set_title("TSLA: ARIMA vs LSTM Forecasts on Test Period")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "05_model_comparison.png"), dpi=130)
    plt.close(fig)

    with open(os.path.join(PROCESSED_DIR, "task2_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # persist artifacts needed by task 3
    np.save(os.path.join(PROCESSED_DIR, "arima_forecast_test.npy"), arima_fc)
    import joblib
    joblib.dump(arima_model, os.path.join(PROCESSED_DIR, "arima_model.pkl"))
    lstm_model.save(os.path.join(PROCESSED_DIR, "lstm_model.keras"))
    joblib.dump(scaler, os.path.join(PROCESSED_DIR, "lstm_scaler.pkl"))

    print(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    run()