"""
data_processing_v2.py
=====================
Wind Power Forecasting — Data Processing with ARIMA Residual Feature

KEY CHANGE vs data_processing.py:
  ARIMA(3,1,2) is fit on the training power series and its residuals
  are computed for all three splits. These residuals are added as an
  explicit feature ("arima_residual") in the scaled dataframe BEFORE
  sequences are built.

  This means the LSTM sees, at every timestep in its input window,
  what ARIMA got wrong recently — giving it direct signal to learn
  the nonlinear correction. The LSTM target remains scaled power_kw,
  and the final hybrid prediction is:

      hybrid = arima_forecast + (lstm_pred_kw - arima_forecast_kw)
             = lstm_pred_kw   (effectively)

  But because ARIMA handles the linear backbone and the LSTM only
  needs to correct the residual signal encoded in its features,
  the decomposition still holds — and alignment is exact because
  residuals are computed and inserted row-by-row before sequencing.

Run this FIRST, then run Model_training_v2.py.
"""

import numpy as np
import pandas as pd
import joblib
import os

from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler


DATA_PATH  = "C:\\Users\\Faisal\\Downloads\\SCADA dataset\\T1.csv"
OUTPUT_DIR = "C:\\Users\\Faisal\\Downloads\\wind_forecasting\\processed\\"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD & CLEAN
# ============================================================

def load_raw(path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["Date/Time"], format="%d %m %Y %H:%M")
    df = df.drop(columns=["Date/Time"])
    df = df.set_index("timestamp").sort_index()
    df = df.rename(columns={
        "LV ActivePower (kW)":            "power_kw",
        "Wind Speed (m/s)":               "wind_speed",
        "Theoretical_Power_Curve (KWh)":  "theoretical_power",
        "Wind Direction (°)":             "wind_dir"
    })
    df["power_kw"] = df["power_kw"].clip(lower=0)
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="10min")
    df = df.reindex(full_idx)
    df = df.ffill(limit=3)
    df = df.dropna()
    print(f"[LOAD] {len(df)} rows")
    return df


# ============================================================
# 2. FEATURE ENGINEERING  (unchanged from original)
# ============================================================

def add_features(df):
    df = df.copy()
    rad = np.deg2rad(df["wind_dir"])
    df["wind_u"] = df["wind_speed"] * np.sin(rad)
    df["wind_v"] = df["wind_speed"] * np.cos(rad)

    df["hour_sin"]  = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df.index.hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df.index.dayofweek / 7)

    for lag in [1, 3, 6, 36, 144]:
        df[f"power_lag_{lag}"] = df["power_kw"].shift(lag)
        df[f"speed_lag_{lag}"] = df["wind_speed"].shift(lag)

    for window, label in [(18, "3h"), (36, "6h")]:
        df[f"power_rollmean_{label}"] = df["power_kw"].shift(1).rolling(window).mean()
        df[f"power_rollstd_{label}"]  = df["power_kw"].shift(1).rolling(window).std()
        df[f"speed_rollmean_{label}"] = df["wind_speed"].shift(1).rolling(window).mean()

    df["power_ratio"] = (df["power_kw"] / (df["theoretical_power"] + 1e-6)).clip(0, 1.5)
    df = df.dropna()
    print(f"[FEATURES] {df.shape[1]} columns, {len(df)} rows")
    return df


# ============================================================
# 3. SPLIT
# ============================================================

def split(df, train=0.7, val=0.15):
    n       = len(df)
    i_train = int(n * train)
    i_val   = int(n * (train + val))
    train_df = df.iloc[:i_train].copy()
    val_df   = df.iloc[i_train:i_val].copy()
    test_df  = df.iloc[i_val:].copy()
    print(f"[SPLIT] train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    return train_df, val_df, test_df


# ============================================================
# 4. ARIMA RESIDUALS AS FEATURE  ← key change
#
#  Fit ARIMA on unscaled training power. Compute in-sample
#  residuals for train, out-of-sample for val+test.
#  Insert as "arima_residual" column before scaling so the
#  scaler normalises it consistently with all other features.
# ============================================================

def add_arima_residual_feature(train_df, val_df, test_df):
    print("\n[ARIMA] Fitting ARIMA(3,1,2) on training power series...")

    train_power = train_df["power_kw"].values
    val_power   = val_df["power_kw"].values
    test_power  = test_df["power_kw"].values

    arima_fit        = ARIMA(train_power, order=(3, 1, 2)).fit()
    train_fitted     = arima_fit.fittedvalues           # in-sample, same length as train

    n_ahead          = len(val_power) + len(test_power)
    forecast         = arima_fit.forecast(steps=n_ahead)
    val_forecast     = forecast[:len(val_power)]
    test_forecast    = forecast[len(val_power):]

    # Residual = actual - ARIMA forecast (what ARIMA got wrong)
    train_residuals  = train_power - train_fitted
    val_residuals    = val_power   - val_forecast
    test_residuals   = test_power  - test_forecast

    train_df = train_df.copy()
    val_df   = val_df.copy()
    test_df  = test_df.copy()

    train_df["arima_residual"] = train_residuals
    val_df["arima_residual"]   = val_residuals
    test_df["arima_residual"]  = test_residuals

    # Save ARIMA forecasts separately — needed in Model_training_v2.py
    np.save(OUTPUT_DIR + "arima_train_fitted.npy", train_fitted)
    np.save(OUTPUT_DIR + "arima_val_forecast.npy",  val_forecast)
    np.save(OUTPUT_DIR + "arima_test_forecast.npy", test_forecast)

    print(f"[ARIMA] Train residuals — mean: {train_residuals.mean():.2f} kW  "
          f"std: {train_residuals.std():.2f} kW")
    print(f"[ARIMA] ARIMA forecasts saved.")

    return train_df, val_df, test_df


# ============================================================
# 5. SCALE  (fit on train only, apply to val/test)
# ============================================================

SKIP_SCALE = {
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "dow_sin", "dow_cos"
}


def fit_and_scale(train_df, val_df, test_df):
    scale_cols = [c for c in train_df.columns if c not in SKIP_SCALE]

    scaler = MinMaxScaler()
    train_scaled = train_df.copy()
    val_scaled   = val_df.copy()
    test_scaled  = test_df.copy()

    train_scaled[scale_cols] = scaler.fit_transform(train_df[scale_cols])
    val_scaled[scale_cols]   = scaler.transform(val_df[scale_cols])
    test_scaled[scale_cols]  = scaler.transform(test_df[scale_cols])

    joblib.dump(scaler, OUTPUT_DIR + "scaler_v2.pkl")
    print(f"\n[SCALER] Fit on {len(scale_cols)} columns. Saved scaler_v2.pkl")
    return train_scaled, val_scaled, test_scaled, scaler


# ============================================================
# 6. SEQUENCES
#    X: all features EXCEPT power_kw (including arima_residual)
#    y: scaled power_kw
# ============================================================

def make_sequences(df, target="power_kw", seq_len=36):
    target_idx   = list(df.columns).index(target)
    feature_idx  = [i for i, c in enumerate(df.columns) if c != target]
    values       = df.values
    X, y         = [], []
    for i in range(seq_len, len(values)):
        X.append(values[i - seq_len:i, feature_idx])
        y.append(values[i, target_idx])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    print(f"[SEQUENCES] X={X.shape}  y={y.shape}")
    return X, y


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    SEQ_LEN = 36

    raw      = load_raw(DATA_PATH)
    featured = add_features(raw)

    train_df, val_df, test_df = split(featured)

    # Insert ARIMA residuals as a feature before scaling
    train_df, val_df, test_df = add_arima_residual_feature(train_df, val_df, test_df)

    # Scale (arima_residual is now included and scaled consistently)
    train_s, val_s, test_s, scaler = fit_and_scale(train_df, val_df, test_df)

    # Save unscaled CSVs for ARIMA re-use in training script
    train_df.to_csv(OUTPUT_DIR + "train_v2.csv")
    val_df.to_csv(OUTPUT_DIR + "val_v2.csv")
    test_df.to_csv(OUTPUT_DIR + "test_v2.csv")

    # Build and save sequences
    X_train, y_train = make_sequences(train_s, "power_kw", SEQ_LEN)
    X_val,   y_val   = make_sequences(val_s,   "power_kw", SEQ_LEN)
    X_test,  y_test  = make_sequences(test_s,  "power_kw", SEQ_LEN)

    np.save(OUTPUT_DIR + "X_train_v2.npy", X_train)
    np.save(OUTPUT_DIR + "y_train_v2.npy", y_train)
    np.save(OUTPUT_DIR + "X_val_v2.npy",   X_val)
    np.save(OUTPUT_DIR + "y_val_v2.npy",   y_val)
    np.save(OUTPUT_DIR + "X_test_v2.npy",  X_test)
    np.save(OUTPUT_DIR + "y_test_v2.npy",  y_test)

    print("\n[DONE] All files saved to", OUTPUT_DIR)
    print("Feature count per timestep:", X_train.shape[2],
          "(was 27, now 28 with arima_residual)")