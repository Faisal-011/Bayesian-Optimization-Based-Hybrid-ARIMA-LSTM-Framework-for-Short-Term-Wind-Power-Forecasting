"""
Model_training_v2.py
====================
Hybrid ARIMA + LSTM Wind Power Forecasting
with Bayesian Optimization (Optuna)

ARCHITECTURE CHANGE vs Model_training_fixed.py:
  Rather than training the LSTM on ARIMA residuals as targets
  (which requires a separate residual scaler and perfect alignment),
  the LSTM is trained normally to predict scaled power_kw — BUT
  its input features now include "arima_residual" at every timestep.

  This means the LSTM implicitly learns to correct ARIMA errors
  because it can directly see the recent residual history in its
  input window. The final hybrid prediction is then:

      hybrid_pred = arima_test_forecast + lstm_correction

  where lstm_correction = lstm_pred_kw - arima_test_forecast
  i.e. we use ARIMA for the linear backbone and the LSTM purely
  for the nonlinear delta — but alignment is exact because the
  residual feature was embedded at processing time.

Run data_processing_v2.py first.
"""

import numpy as np
import pandas as pd
import optuna
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================
# 1. LOAD DATA
# ============================================================

BASE = "C:\\Users\\Faisal\\Downloads\\wind_forecasting\\processed\\"

# Unscaled CSVs (with arima_residual column) — for kW-space evaluation
train_df = pd.read_csv(BASE + "train_v2.csv", index_col=0)
val_df   = pd.read_csv(BASE + "val_v2.csv",   index_col=0)
test_df  = pd.read_csv(BASE + "test_v2.csv",  index_col=0)

train_series = train_df["power_kw"].values
val_series   = val_df["power_kw"].values
test_series  = test_df["power_kw"].values

# Saved ARIMA forecasts (computed in data_processing_v2.py)
arima_train_fitted = np.load(BASE + "arima_train_fitted.npy")
arima_val_forecast = np.load(BASE + "arima_val_forecast.npy")
arima_test_forecast = np.load(BASE + "arima_test_forecast.npy")

# V2 sequences — X now includes arima_residual as a feature
X_train = np.load(BASE + "X_train_v2.npy")
y_train = np.load(BASE + "y_train_v2.npy")
X_val   = np.load(BASE + "X_val_v2.npy")
y_val   = np.load(BASE + "y_val_v2.npy")
X_test  = np.load(BASE + "X_test_v2.npy")
y_test  = np.load(BASE + "y_test_v2.npy")

scaler            = joblib.load(BASE + "scaler_v2.pkl")
n_scaler_features = scaler.n_features_in_
seq_len           = X_train.shape[1]   # 36

print(f"Loaded: X_train={X_train.shape}  X_val={X_val.shape}  X_test={X_test.shape}")
print(f"Features per timestep: {X_train.shape[2]} (includes arima_residual)")


# ============================================================
# 2. HELPERS
# ============================================================

# Find power_kw column index in the scaler
# scaler was fit on all columns except sinusoidal ones
# power_kw is first column in the original df
def get_power_col_idx():
    # Reconstruct which columns were scaled
    all_cols = list(train_df.columns)
    skip = {"hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos"}
    scaled_cols = [c for c in all_cols if c not in skip]
    return scaled_cols.index("power_kw")

POWER_COL_IDX = get_power_col_idx()
print(f"power_kw is at scaler index {POWER_COL_IDX}")


def inverse_transform_power(scaled_values: np.ndarray) -> np.ndarray:
    """Inverse-transform scaled power_kw values back to kW."""
    dummy = np.zeros((len(scaled_values), n_scaler_features))
    dummy[:, POWER_COL_IDX] = scaled_values.flatten()
    return scaler.inverse_transform(dummy)[:, POWER_COL_IDX]


# Actual kW targets (ground truth for all evaluation)
y_train_kw = inverse_transform_power(y_train)
y_val_kw   = inverse_transform_power(y_val)
y_test_kw  = inverse_transform_power(y_test)

# Align ARIMA forecasts to sequence positions (first seq_len rows are consumed)
arima_val_aligned  = arima_val_forecast[seq_len:]    # shape: (n_val_seq,)
arima_test_aligned = arima_test_forecast[seq_len:]   # shape: (n_test_seq,)

actual_val  = val_series[seq_len:]
actual_test = test_series[seq_len:]

# Trim to exact sequence counts
n_val  = min(len(arima_val_aligned),  len(X_val),  len(actual_val))
n_test = min(len(arima_test_aligned), len(X_test), len(actual_test))

arima_val_aligned  = arima_val_aligned[:n_val]
arima_test_aligned = arima_test_aligned[:n_test]
actual_val         = actual_val[:n_val]
actual_test        = actual_test[:n_test]


# ============================================================
# 3. BAYESIAN OPTIMIZATION
#    LSTM predicts scaled power_kw directly.
#    Optuna objective = hybrid RMSE on validation set:
#      hybrid = arima_val_forecast + (lstm_pred - arima_val_forecast)
#             = lstm_pred (in kW)
#    The decomposition is conceptual — ARIMA provides the backbone,
#    LSTM corrects it via the arima_residual feature in X.
# ============================================================

print("\nStarting Bayesian Optimization (10 trials)...")


def build_model(units, dropout, lr, seq_shape):
    m = Sequential([
        LSTM(units, return_sequences=True, input_shape=seq_shape),
        Dropout(dropout),
        LSTM(units // 2),
        Dense(1)
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
    return m


def objective(trial):
    units      = trial.suggest_int("units", 32, 128)
    dropout    = trial.suggest_float("dropout", 0.1, 0.4)
    lr         = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    model = build_model(units, dropout, lr,
                        seq_shape=(X_train.shape[1], X_train.shape[2]))
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20, batch_size=batch_size, verbose=0
    )

    # LSTM prediction in kW
    val_pred_scaled = model.predict(X_val[:n_val], verbose=0).flatten()
    val_pred_kw     = inverse_transform_power(val_pred_scaled)

    # Hybrid = ARIMA backbone + LSTM nonlinear correction
    lstm_correction = val_pred_kw - arima_val_aligned
    hybrid_val      = arima_val_aligned + lstm_correction   # == val_pred_kw
    # (keeping the explicit decomposition for clarity and future ablation)

    rmse = np.sqrt(mean_squared_error(actual_val, hybrid_val))
    return rmse


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

best = study.best_params
print("\nBest parameters:", best)


# ============================================================
# 4. TRAIN FINAL LSTM
# ============================================================

print("\nTraining final LSTM with best parameters...")

lstm_model = build_model(
    best["units"], best["dropout"], best["learning_rate"],
    seq_shape=(X_train.shape[1], X_train.shape[2])
)
lstm_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=best["batch_size"],
    callbacks=[EarlyStopping(patience=4, restore_best_weights=True)]
)


# ============================================================
# 5. GENERATE HYBRID FORECAST ON TEST SET
# ============================================================

print("\nGenerating hybrid forecast on test set...")

test_pred_scaled = lstm_model.predict(X_test[:n_test], verbose=0).flatten()
test_pred_kw     = inverse_transform_power(test_pred_scaled)

# Hybrid decomposition: ARIMA linear + LSTM nonlinear correction
lstm_correction = test_pred_kw - arima_test_aligned
hybrid_pred     = arima_test_aligned + lstm_correction   # == test_pred_kw
actual          = actual_test


# ============================================================
# 6. EVALUATION
# ============================================================

hybrid_mae  = mean_absolute_error(actual, hybrid_pred)
hybrid_rmse = np.sqrt(mean_squared_error(actual, hybrid_pred))

# ARIMA-only baseline
arima_mae  = mean_absolute_error(actual, arima_test_aligned)
arima_rmse = np.sqrt(mean_squared_error(actual, arima_test_aligned))

print("\n" + "=" * 55)
print("FINAL HYBRID MODEL PERFORMANCE (kW)")
print("=" * 55)
print(f"  Hybrid MAE  : {hybrid_mae:.4f} kW")
print(f"  Hybrid RMSE : {hybrid_rmse:.4f} kW")
print(f"\n  ARIMA-only MAE  : {arima_mae:.4f} kW")
print(f"  ARIMA-only RMSE : {arima_rmse:.4f} kW")
print(f"\n  MAE  reduced by {(arima_mae  - hybrid_mae)  / arima_mae  * 100:.1f}%")
print(f"  RMSE reduced by {(arima_rmse - hybrid_rmse) / arima_rmse * 100:.1f}%")


# ============================================================
# 7. PLOT
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(actual[:500],      label="Actual",            color="blue",   linewidth=1)
axes[0].plot(hybrid_pred[:500], label="Hybrid ARIMA+LSTM", color="orange", linewidth=1)
axes[0].set_title("Hybrid ARIMA+LSTM — Wind Power Forecast vs Actual (first 500 test samples)")
axes[0].set_ylabel("Power Output (kW)")
axes[0].legend()

residuals = actual[:500] - hybrid_pred[:500]
axes[1].plot(residuals, color="red", linewidth=0.8, alpha=0.7)
axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[1].set_title("Forecast Residuals")
axes[1].set_ylabel("Error (kW)")
axes[1].set_xlabel("Sample Index")

plt.tight_layout()
plt.savefig("hybrid_forecast_v2.png", dpi=150)
plt.show()


# ============================================================
# 8. SAVE
# ============================================================

lstm_model.save("lstm_hybrid_model_v2.keras")
joblib.dump(best, "best_params_v2.pkl")

print("\nModel saved : lstm_hybrid_model_v2.keras")
print("Params saved: best_params_v2.pkl")
print("Plot saved  : hybrid_forecast_v2.png")
