"""
logic.py - Hybrid ML Prediction Engine
=======================================
Research: ML-Driven Autoscaling with Waterfall Dependency Logic
Models:
  1. ARIMA         - Stable, linear traffic (low volatility)
  2. Random Forest - Flash Crowd spikes    (high volatility)
"""

import numpy as np
import pickle
import os
import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────
RF_MODEL_DIR         = "models"          # Folder for per-service RF models
SEQUENCE_LENGTH      = 10
VOLATILITY_THRESHOLD = 15.0
SCALE_OUT_THRESHOLD  = 70.0
SCALE_IN_THRESHOLD   = 35.0
RETRAIN_INTERVAL     = 50

os.makedirs(RF_MODEL_DIR, exist_ok=True)

_retrain_counters = {}   # Per-service retrain counter


# ── Feature Engineering ───────────────────────────────────
def build_features(cpu_history, network_history):
    cpu_arr = np.array(cpu_history[-SEQUENCE_LENGTH:], dtype=np.float32)
    net_val = float(network_history[-1]) if network_history else 0.0

    if len(cpu_arr) < SEQUENCE_LENGTH:
        cpu_arr = np.pad(cpu_arr, (SEQUENCE_LENGTH - len(cpu_arr), 0))

    cpu_mean = float(np.mean(cpu_arr))
    cpu_std  = float(np.std(cpu_arr))
    cpu_min  = float(np.min(cpu_arr))
    cpu_max  = float(np.max(cpu_arr))
    slope    = float(np.polyfit(range(5), cpu_arr[-5:], 1)[0]) if len(cpu_arr) >= 5 else 0.0

    features = list(cpu_arr) + [cpu_mean, cpu_std, cpu_min, cpu_max, net_val, slope]
    return np.array(features, dtype=np.float32).reshape(1, -1)


def prepare_training_data(cpu_history, network_history):
    X, y    = [], []
    cpu_list = list(cpu_history)
    net_list = list(network_history)
    min_len  = min(len(cpu_list), len(net_list))

    for i in range(SEQUENCE_LENGTH, min_len - 1):
        features = build_features(cpu_list[i - SEQUENCE_LENGTH:i],
                                  net_list[i - SEQUENCE_LENGTH:i])
        X.append(features.flatten())
        y.append(cpu_list[i])

    return np.array(X), np.array(y)


# ── Random Forest Training ────────────────────────────────
def train_random_forest(service, cpu_history, network_history):
    if len(cpu_history) < SEQUENCE_LENGTH + 5:
        print(f"[RF:{service}] Not enough data ({len(cpu_history)} pts).")
        return None

    print(f"[RF:{service}] Training...")
    X, y = prepare_training_data(cpu_history, network_history)

    if len(X) == 0:
        return None

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    path = os.path.join(RF_MODEL_DIR, f"rf_{service}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)

    print(f"[RF:{service}] Saved to {path}")
    return model


# ── Random Forest Prediction ──────────────────────────────
def predict_with_random_forest(service, cpu_history, network_history):
    path = os.path.join(RF_MODEL_DIR, f"rf_{service}.pkl")
    if not os.path.exists(path):
        return None

    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        features   = build_features(cpu_history, network_history)
        prediction = float(model.predict(features)[0])
        return round(max(0.0, min(100.0, prediction)), 2)
    except Exception as e:
        print(f"[RF:{service}] Prediction error: {e}")
        return None


# ── ARIMA Prediction ──────────────────────────────────────
def predict_with_arima(service, cpu_history):
    try:
        model  = ARIMA(cpu_history, order=(2, 1, 0)).fit()
        result = model.forecast(steps=1).iloc[0]
        return max(0.0, round(float(result), 2))
    except Exception as e:
        print(f"[ARIMA:{service}] Error: {e}")
        return float(cpu_history[-1])


# ── Hybrid Adaptive Toggle ────────────────────────────────
def adaptive_predict(service, cpu_history, network_history):
    """
    Selects ARIMA or Random Forest based on traffic volatility.
    Called independently for each service (frontend / backend).
    """
    global _retrain_counters

    if len(cpu_history) < 10:
        print(f"[{service}] Collecting data ({len(cpu_history)}/10)...")
        return float(cpu_history[-1])

    volatility = float(np.std(list(network_history)[-5:]))
    print(f"[{service}] Volatility: {volatility:.2f} | Threshold: {VOLATILITY_THRESHOLD}")

    # Periodic retraining
    _retrain_counters[service] = _retrain_counters.get(service, 0) + 1
    if _retrain_counters[service] >= RETRAIN_INTERVAL:
        train_random_forest(service, list(cpu_history), list(network_history))
        _retrain_counters[service] = 0

    if volatility < VOLATILITY_THRESHOLD:
        pred = predict_with_arima(service, list(cpu_history))
        print(f"[ARIMA:{service}] Stable. Predicted CPU: {pred}%")
        return pred
    else:
        rf_pred = predict_with_random_forest(service, list(cpu_history), list(network_history))
        if rf_pred is not None:
            print(f"[RF:{service}] Flash Crowd! Predicted CPU: {rf_pred}%")
            return rf_pred
        # RF not ready — ARIMA with safety buffer
        arima_pred = predict_with_arima(service, list(cpu_history))
        buffered   = min(100.0, arima_pred * 1.2)
        print(f"[ARIMA+Buffer:{service}] RF not ready. Buffered: {buffered}%")
        return buffered


# ── Smart Scale-In: Anti-Flapping ────────────────────────
def calculate_target_capacity(service, predicted_cpu, current_capacity):
    """
    Scale-Out: adds instance when CPU spike predicted.
    Scale-In:  redistribution check prevents flapping.
    """
    if predicted_cpu > SCALE_OUT_THRESHOLD:
        new = current_capacity + 1
        print(f"[Scaler:{service}] SCALE OUT {current_capacity} -> {new} ({predicted_cpu}%)")
        return new

    elif predicted_cpu < SCALE_IN_THRESHOLD and current_capacity > 1:
        proposed      = current_capacity - 1
        redistributed = (predicted_cpu * current_capacity) / proposed

        if redistributed >= SCALE_OUT_THRESHOLD:
            print(f"[Anti-Flapping:{service}] Aborted. Redistributed CPU: {redistributed:.1f}%")
            return current_capacity

        print(f"[Scaler:{service}] SCALE IN {current_capacity} -> {proposed} "
              f"(redistributed: {redistributed:.1f}%)")
        return proposed

    print(f"[Scaler:{service}] STABLE at {current_capacity} ({predicted_cpu}%)")
    return current_capacity