"""
Hybrid ARIMA + LSTM Wind Power Forecasting
with Bayesian Optimization (Optuna)
- Fixed unit mismatch bug
- Fixed ARIMA forecast alignment
"""

import numpy as np
import pandas as pd
import optuna
import joblib
import tensorflow as tf

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================
# 1. LOAD DATA
# ============================================================

BASE = "C:\\Users\\Faisal\\Downloads\\wind_forecasting\\processed\\"

# Unscaled CSVs — used for ARIMA (operates in real kW space)
train_df = pd.read_csv(BASE + "train.csv", index_col=0)
val_df   = pd.read_csv(BASE + "val.csv",   index_col=0)
test_df  = pd.read_csv(BASE + "test.csv",  index_col=0)

train_series = train_df["power_kw"].values
val_series   = val_df["power_kw"].values
test_series  = test_df["power_kw"].values

# Scaled sequences — used for LSTM (operates in [0,1] space)
X_train = np.load(BASE + "X_train_seq.npy")
y_train = np.load(BASE + "y_train_seq.npy")
X_val   = np.load(BASE + "X_val_seq.npy")
y_val   = np.load(BASE + "y_val_seq.npy")
X_test  = np.load(BASE + "X_test_seq.npy")
y_test  = np.load(BASE + "y_test_seq.npy")

# Load scaler to inverse-transform LSTM predictions back to kW
scaler = joblib.load(BASE + "scaler.pkl")

# Detect scaler's expected feature count and power_kw column index
n_scaler_features = scaler.n_features_in_
POWER_COL_IDX = 0  # power_kw is first column in the scaled CSV


def inverse_transform_power(scaled_values: np.ndarray) -> np.ndarray:
    """
    Inverse-transform a 1D array of scaled power_kw values back to kW.
    Pads with zeros for all other columns to match scaler's expected shape.
    """
    dummy = np.zeros((len(scaled_values), n_scaler_features))
    dummy[:, POWER_COL_IDX] = scaled_values.flatten()
    return scaler.inverse_transform(dummy)[:, POWER_COL_IDX]


# ============================================================
# 2. ARIMA MODEL (LINEAR COMPONENT)
# ============================================================

print("Training ARIMA(3,1,2) on training series...")

arima_model = ARIMA(train_series, order=(3, 1, 2))
arima_fit   = arima_model.fit()
print(arima_fit.summary())

# In-sample fitted values for training set (used to compute residuals)
arima_train_fitted = arima_fit.fittedvalues          # shape: (len(train_series),)

# Out-of-sample forecast for validation + test (needed for hybrid combination)
n_forecast = len(val_series) + len(test_series)
arima_forecast_all = arima_fit.forecast(steps=n_forecast)

arima_val_pred  = np.array(arima_forecast_all[:len(val_series)])
arima_test_pred = np.array(arima_forecast_all[len(val_series):])


# ============================================================
# 3. RESIDUAL COMPUTATION  (in kW — unscaled space)
# ============================================================

print("\nComputing ARIMA residuals...")

# Residuals = what ARIMA could NOT explain — LSTM will learn these
# seq_len=36 rows are dropped at start of each split during sequencing,
# so align residuals to match y_train length
seq_len = X_train.shape[1]

train_residuals_kw = train_series - arima_train_fitted   # full training residuals in kW

# y_train (scaled) corresponds to rows [seq_len:] of train_df
# Convert y_train back to kW so LSTM targets are in the same space as ARIMA residuals
y_train_kw = inverse_transform_power(y_train)
y_val_kw   = inverse_transform_power(y_val)
y_test_kw  = inverse_transform_power(y_test)

# ARIMA residuals aligned to sequence positions
train_res_aligned = train_residuals_kw[seq_len:]        # shape matches y_train_kw

# Val residuals: ARIMA val forecast vs actual val targets
val_res_aligned = val_series[seq_len:] - arima_val_pred[seq_len:]


# ============================================================
# 4. BAYESIAN OPTIMIZATION (OPTUNA)
# ============================================================

print("\nStarting Bayesian Optimization (10 trials)...")

def objective(trial):
    units      = trial.suggest_int("units", 32, 128)
    dropout    = trial.suggest_float("dropout", 0.1, 0.4)
    lr         = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    model = Sequential([
        LSTM(units, return_sequences=True,
             input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(dropout),
        LSTM(units // 2),
        Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=batch_size,
        verbose=0
    )

    # Evaluate on validation set in kW space for a meaningful objective
    val_pred_scaled = model.predict(X_val, verbose=0).flatten()
    val_pred_kw     = inverse_transform_power(val_pred_scaled)

    # LSTM predicts the residual; hybrid val prediction = ARIMA val + LSTM residual
    hybrid_val = arima_val_pred[seq_len:] + val_pred_kw
    actual_val = val_series[seq_len:]

    rmse = np.sqrt(mean_squared_error(actual_val, hybrid_val))
    return rmse


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

best = study.best_params
print("\nBest parameters:", best)


# ============================================================
# 5. TRAIN FINAL LSTM WITH BEST PARAMETERS
# ============================================================

print("\nTraining final LSTM with best parameters...")

lstm_model = Sequential([
    LSTM(best["units"], return_sequences=True,
         input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(best["dropout"]),
    LSTM(best["units"] // 2),
    Dense(1)
])

lstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(best["learning_rate"]),
    loss="mse",
    metrics=["mae"]
)

lstm_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=best["batch_size"]
)


# ============================================================
# 6. GENERATE HYBRID FORECAST ON TEST SET
# ============================================================

print("\nGenerating hybrid forecast on test set...")

# LSTM predicts scaled residuals → inverse transform to kW
lstm_test_pred_scaled = lstm_model.predict(X_test).flatten()
lstm_test_pred_kw     = inverse_transform_power(lstm_test_pred_scaled)

# Align ARIMA test forecast to sequence length
arima_test_aligned = arima_test_pred[seq_len:]          # drop first seq_len steps
actual_test        = test_series[seq_len:]              # same alignment

# Final hybrid = ARIMA linear component + LSTM residual component
n = min(len(arima_test_aligned), len(lstm_test_pred_kw), len(actual_test))
hybrid_pred = arima_test_aligned[:n] + lstm_test_pred_kw[:n]
actual      = actual_test[:n]


# ============================================================
# 7. EVALUATION
# ============================================================

mae  = mean_absolute_error(actual, hybrid_pred)
rmse = np.sqrt(mean_squared_error(actual, hybrid_pred))

print("\n" + "="*50)
print("FINAL HYBRID MODEL PERFORMANCE (kW)")
print("="*50)
print(f"MAE  : {mae:.4f} kW")
print(f"RMSE : {rmse:.4f} kW")

# Also report ARIMA-only baseline for comparison
arima_only_mae  = mean_absolute_error(actual, arima_test_aligned[:n])
arima_only_rmse = np.sqrt(mean_squared_error(actual, arima_test_aligned[:n]))
print(f"\nARIMA-only baseline:")
print(f"MAE  : {arima_only_mae:.4f} kW")
print(f"RMSE : {arima_only_rmse:.4f} kW")

improvement_mae  = (arima_only_mae  - mae)  / arima_only_mae  * 100
improvement_rmse = (arima_only_rmse - rmse) / arima_only_rmse * 100
print(f"\nHybrid improvement over ARIMA:")
print(f"MAE  reduced by {improvement_mae:.1f}%")
print(f"RMSE reduced by {improvement_rmse:.1f}%")


# ============================================================
# 8. PLOT
# ============================================================

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Top: hybrid vs actual
axes[0].plot(actual[:500],      label="Actual",            color="blue",   linewidth=1)
axes[0].plot(hybrid_pred[:500], label="Hybrid ARIMA+LSTM", color="orange", linewidth=1)
axes[0].set_title("Hybrid ARIMA+LSTM — Wind Power Forecast vs Actual (first 500 test samples)")
axes[0].set_ylabel("Power Output (kW)")
axes[0].legend()

# Bottom: residuals
residuals = actual[:500] - hybrid_pred[:500]
axes[1].plot(residuals, color="red", linewidth=0.8, alpha=0.7)
axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[1].set_title("Forecast Residuals")
axes[1].set_ylabel("Error (kW)")
axes[1].set_xlabel("Sample Index")

plt.tight_layout()
plt.savefig("hybrid_forecast.png", dpi=150)
plt.show()


# ============================================================
# 9. SAVE
# ============================================================

lstm_model.save("lstm_hybrid_model.h5")
print("\nModel saved: lstm_hybrid_model.h5")
print("Plot saved:  hybrid_forecast.png")