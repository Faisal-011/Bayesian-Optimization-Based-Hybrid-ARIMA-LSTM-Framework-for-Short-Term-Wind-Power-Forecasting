"""
baselines_v2.py
===============
Baseline comparison for:
    "Bayesian Optimization-Based Hybrid ARIMA-LSTM Framework
     for Short-Term Wind Power Forecasting"

Uses the v2 processed sequences (which include arima_residual as a feature).
Trains three baselines on the same v2 sequences for a fair comparison:
    1. Standalone LSTM
    2. XGBoost
    3. Lightweight Transformer

Hybrid ARIMA-LSTM results are loaded from the saved v2 model.

Run data_processing_v2.py and Model_training_v2.py before this script.
"""

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, MultiHeadAttention,
    LayerNormalization, GlobalAveragePooling1D, Input
)
from tensorflow.keras.callbacks import EarlyStopping
from xgboost import XGBRegressor
import matplotlib.pyplot as plt


# ============================================================
# 0. CONFIGURATION
# ============================================================

BASE    = "C:\\Users\\Faisal\\Downloads\\wind_forecasting\\processed\\"
SEQ_LEN = 36


# ============================================================
# 1. LOAD V2 ARTIFACTS
# ============================================================

print("Loading v2 artifacts...")

train_df = pd.read_csv(BASE + "train_v2.csv", index_col=0)
val_df   = pd.read_csv(BASE + "val_v2.csv",   index_col=0)
test_df  = pd.read_csv(BASE + "test_v2.csv",  index_col=0)

train_series = train_df["power_kw"].values
val_series   = val_df["power_kw"].values
test_series  = test_df["power_kw"].values

# V2 sequences — include arima_residual feature
X_train = np.load(BASE + "X_train_v2.npy")
y_train = np.load(BASE + "y_train_v2.npy")
X_val   = np.load(BASE + "X_val_v2.npy")
y_val   = np.load(BASE + "y_val_v2.npy")
X_test  = np.load(BASE + "X_test_v2.npy")
y_test  = np.load(BASE + "y_test_v2.npy")

# ARIMA forecasts saved by data_processing_v2.py
arima_test_forecast = np.load(BASE + "arima_test_forecast.npy")
arima_test_aligned  = arima_test_forecast[SEQ_LEN:]
actual_test         = test_series[SEQ_LEN:]

scaler            = joblib.load(BASE + "scaler_v2.pkl")
n_scaler_features = scaler.n_features_in_

print(f"  X_train={X_train.shape}  X_test={X_test.shape}")
print(f"  Features per timestep: {X_train.shape[2]}")


# ============================================================
# 2. HELPERS
# ============================================================

def get_power_col_idx():
    skip = {"hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos"}
    scaled_cols = [c for c in train_df.columns if c not in skip]
    return scaled_cols.index("power_kw")

POWER_COL_IDX = get_power_col_idx()

def inv_power(scaled: np.ndarray) -> np.ndarray:
    dummy = np.zeros((len(scaled), n_scaler_features))
    dummy[:, POWER_COL_IDX] = scaled.flatten()
    return scaler.inverse_transform(dummy)[:, POWER_COL_IDX]

y_test_kw = inv_power(y_test)

# ARIMA-only baseline metrics (shared denominator for all reductions)
n_shared       = min(len(arima_test_aligned), len(y_test_kw), len(actual_test))
actual         = actual_test[:n_shared]
arima_aligned  = arima_test_aligned[:n_shared]

arima_mae  = mean_absolute_error(actual, arima_aligned)
arima_rmse = np.sqrt(mean_squared_error(actual, arima_aligned))
print(f"\n  [ARIMA baseline] MAE={arima_mae:.2f} kW  RMSE={arima_rmse:.2f} kW")


# ============================================================
# 3. HYBRID ARIMA-LSTM  (load saved v2 model)
# ============================================================

print("\n[Hybrid] Loading lstm_hybrid_model_v2.keras...")

lstm_hybrid = tf.keras.models.load_model(
    "lstm_hybrid_model_v2.keras", compile=False
)
lstm_hybrid.compile(optimizer="adam", loss="mse")

hybrid_pred_scaled = lstm_hybrid.predict(X_test[:n_shared], verbose=0).flatten()
hybrid_pred_kw     = inv_power(hybrid_pred_scaled)

# Hybrid = ARIMA backbone + LSTM nonlinear correction
lstm_correction = hybrid_pred_kw - arima_aligned
hybrid_pred     = arima_aligned + lstm_correction   # == hybrid_pred_kw

hybrid_mae  = mean_absolute_error(actual, hybrid_pred)
hybrid_rmse = np.sqrt(mean_squared_error(actual, hybrid_pred))
print(f"  [Hybrid ARIMA-LSTM] MAE={hybrid_mae:.2f} kW  RMSE={hybrid_rmse:.2f} kW")


# ============================================================
# 4. STANDALONE LSTM
#    Uses v2 sequences (same features as hybrid) but no ARIMA
#    decomposition — predicts power_kw directly.
#    Load best params from Bayesian search for fair comparison.
# ============================================================

print("\n[Standalone LSTM] Training...")

best = joblib.load("best_params_v2.pkl")
UNITS   = best["units"]
DROPOUT = best["dropout"]
LR      = best["learning_rate"]
BATCH   = best["batch_size"]
print(f"  Using Bayesian params: units={UNITS}, dropout={DROPOUT:.2f}, "
      f"lr={LR:.4f}, batch={BATCH}")

standalone_lstm = Sequential([
    LSTM(UNITS, return_sequences=True,
         input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(DROPOUT),
    LSTM(UNITS // 2),
    Dense(1)
])
standalone_lstm.compile(optimizer=tf.keras.optimizers.Adam(LR), loss="mse")
standalone_lstm.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20, batch_size=BATCH,
    callbacks=[EarlyStopping(patience=4, restore_best_weights=True)],
    verbose=1
)

sl_pred_kw = inv_power(standalone_lstm.predict(X_test[:n_shared], verbose=0).flatten())
sl_mae     = mean_absolute_error(actual, sl_pred_kw)
sl_rmse    = np.sqrt(mean_squared_error(actual, sl_pred_kw))
print(f"  [Standalone LSTM] MAE={sl_mae:.2f} kW  RMSE={sl_rmse:.2f} kW")


# ============================================================
# 5. XGBOOST
# ============================================================

print("\n[XGBoost] Training...")

X_train_flat = X_train.reshape(len(X_train), -1)
X_val_flat   = X_val.reshape(len(X_val),   -1)
X_test_flat  = X_test[:n_shared].reshape(n_shared, -1)

xgb = XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1,
    early_stopping_rounds=20, eval_metric="rmse", verbosity=1
)
xgb.fit(X_train_flat, y_train, eval_set=[(X_val_flat, y_val)], verbose=50)

xgb_pred_kw = inv_power(xgb.predict(X_test_flat))
xgb_mae     = mean_absolute_error(actual, xgb_pred_kw)
xgb_rmse    = np.sqrt(mean_squared_error(actual, xgb_pred_kw))
print(f"  [XGBoost] MAE={xgb_mae:.2f} kW  RMSE={xgb_rmse:.2f} kW")


# ============================================================
# 6. TRANSFORMER (encoder-only)
# ============================================================

print("\n[Transformer] Building and training...")

def build_transformer(seq_len, n_features, num_heads=4, ff_dim=64, dropout=0.1):
    inputs   = Input(shape=(seq_len, n_features))
    attn_out = MultiHeadAttention(
        num_heads=num_heads, key_dim=max(1, n_features // num_heads)
    )(inputs, inputs)
    attn_out = Dropout(dropout)(attn_out)
    attn_out = LayerNormalization(epsilon=1e-6)(inputs + attn_out)
    ff_out   = Dense(ff_dim, activation="relu")(attn_out)
    ff_out   = Dense(n_features)(ff_out)
    ff_out   = Dropout(dropout)(ff_out)
    ff_out   = LayerNormalization(epsilon=1e-6)(attn_out + ff_out)
    pooled   = GlobalAveragePooling1D()(ff_out)
    output   = Dense(32, activation="relu")(pooled)
    output   = Dense(1)(output)
    return Model(inputs, output)

transformer = build_transformer(X_train.shape[1], X_train.shape[2])
transformer.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
transformer.summary()
transformer.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20, batch_size=64,
    callbacks=[EarlyStopping(patience=4, restore_best_weights=True)],
    verbose=1
)

tf_pred_kw = inv_power(transformer.predict(X_test[:n_shared], verbose=0).flatten())
tf_mae     = mean_absolute_error(actual, tf_pred_kw)
tf_rmse    = np.sqrt(mean_squared_error(actual, tf_pred_kw))
print(f"  [Transformer] MAE={tf_mae:.2f} kW  RMSE={tf_rmse:.2f} kW")


# ============================================================
# 7. UNIFIED RESULTS TABLE
# ============================================================

def pct(baseline, value):
    return f"{(baseline - value) / baseline * 100:.1f}%"

results = [
    ("ARIMA (3,1,2)",                arima_mae,  arima_rmse),
    ("Standalone LSTM",              sl_mae,     sl_rmse),
    ("XGBoost",                      xgb_mae,    xgb_rmse),
    ("Transformer (encoder-only)",   tf_mae,     tf_rmse),
    ("Hybrid ARIMA-LSTM (Proposed)", hybrid_mae, hybrid_rmse),
]

print("\n" + "=" * 70)
print(f"{'Model':<32} {'MAE (kW)':>10} {'RMSE (kW)':>11} {'MAE Red.':>10} {'RMSE Red.':>11}")
print("-" * 70)
for name, mae, rmse in results:
    if name.startswith("ARIMA (3"):
        print(f"{name:<32} {mae:>10.2f} {rmse:>11.2f} {'—':>10} {'—':>11}")
    else:
        print(f"{name:<32} {mae:>10.2f} {rmse:>11.2f} "
              f"{pct(arima_mae, mae):>10} {pct(arima_rmse, rmse):>11}")
print("=" * 70)
print("Note: reduction % relative to ARIMA baseline.")


# ============================================================
# 8. SAVE CSV
# ============================================================

rows = []
for name, mae, rmse in results:
    rows.append({
        "Model":          name,
        "MAE (kW)":       round(mae, 2),
        "RMSE (kW)":      round(rmse, 2),
        "MAE Reduction":  "—" if name.startswith("ARIMA") else pct(arima_mae, mae),
        "RMSE Reduction": "—" if name.startswith("ARIMA") else pct(arima_rmse, rmse),
    })
pd.DataFrame(rows).to_csv("baseline_comparison_v2_results.csv", index=False)
print("\nResults saved to: baseline_comparison_v2_results.csv")


# ============================================================
# 9. BAR CHART
# ============================================================

model_names = [r[0] for r in results]
mae_vals    = [r[1] for r in results]
rmse_vals   = [r[2] for r in results]

# Highlight the proposed model
colors_mae  = ["#4C72B0" if "Proposed" not in n else "#2ca02c" for n in model_names]
colors_rmse = ["#DD8452" if "Proposed" not in n else "#98df8a" for n in model_names]

x   = np.arange(len(model_names))
w   = 0.35
fig, ax = plt.subplots(figsize=(13, 6))

bars_mae  = ax.bar(x - w/2, mae_vals,  w, label="MAE (kW)",  color=colors_mae,  alpha=0.88)
bars_rmse = ax.bar(x + w/2, rmse_vals, w, label="RMSE (kW)", color=colors_rmse, alpha=0.88)

for bar in bars_mae:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
            f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9)
for bar in bars_rmse:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
            f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=10)
ax.set_ylabel("Error (kW)", fontsize=11)
ax.set_title("Model Comparison — MAE and RMSE on Test Set\n"
             "(green bars = proposed hybrid model)", fontsize=12)
ax.legend(fontsize=10)
ax.set_ylim(0, max(rmse_vals) * 1.18)
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("baseline_comparison_v2.png", dpi=150)
plt.show()
print("Bar chart saved to: baseline_comparison_v2.png")