"""
Wind Power Forecasting — Data Loading & Feature Engineering
Dataset: SCADA Turbine T1 (10-minute resolution)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "T1.csv"
OUTPUT_DIR = "processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD & CLEAN DATA
# ============================================================

def load_raw(path):

    df = pd.read_csv(path)

    df["timestamp"] = pd.to_datetime(df["Date/Time"], format="%d %m %Y %H:%M")
    df = df.drop(columns=["Date/Time"])
    df = df.set_index("timestamp").sort_index()

    df = df.rename(columns={
        "LV ActivePower (kW)": "power_kw",
        "Wind Speed (m/s)": "wind_speed",
        "Theoretical_Power_Curve (KWh)": "theoretical_power",
        "Wind Direction (°)": "wind_dir"
    })

    df["power_kw"] = df["power_kw"].clip(lower=0)

    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="10min")
    df = df.reindex(full_idx)

    df = df.ffill(limit=3)
    df = df.dropna()

    print(f"[LOAD] {len(df)} rows loaded")

    return df


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def add_features(df):

    df = df.copy()

    # Wind direction → vector components
    rad = np.deg2rad(df["wind_dir"])

    df["wind_u"] = df["wind_speed"] * np.sin(rad)
    df["wind_v"] = df["wind_speed"] * np.cos(rad)

    # Time encodings
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)

    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    # Lag features
    lag_steps = [1, 3, 6, 36, 144]

    for lag in lag_steps:

        df[f"power_lag_{lag}"] = df["power_kw"].shift(lag)
        df[f"speed_lag_{lag}"] = df["wind_speed"].shift(lag)

    # Rolling statistics
    for window, label in [(18, "3h"), (36, "6h")]:

        df[f"power_rollmean_{label}"] = df["power_kw"].shift(1).rolling(window).mean()
        df[f"power_rollstd_{label}"] = df["power_kw"].shift(1).rolling(window).std()

        df[f"speed_rollmean_{label}"] = df["wind_speed"].shift(1).rolling(window).mean()

    # Power curve ratio
    df["power_ratio"] = (df["power_kw"] / (df["theoretical_power"] + 1e-6)).clip(0, 1.5)

    df = df.dropna()

    print(f"[FEATURES] {df.shape[1]} columns")

    return df


# ============================================================
# 3. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def split(df, train=0.7, val=0.15):

    n = len(df)

    i_train = int(n * train)
    i_val = int(n * (train + val))

    train_df = df.iloc[:i_train]
    val_df = df.iloc[i_train:i_val]
    test_df = df.iloc[i_val:]

    print(f"[SPLIT] train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    return train_df, val_df, test_df


# ============================================================
# 4. SCALING
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
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()

    train_scaled[scale_cols] = scaler.fit_transform(train_df[scale_cols])
    val_scaled[scale_cols] = scaler.transform(val_df[scale_cols])
    test_scaled[scale_cols] = scaler.transform(test_df[scale_cols])

    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

    print("[SCALER] saved")

    return train_scaled, val_scaled, test_scaled


# ============================================================
# 5. CREATE LSTM SEQUENCES
# ============================================================

def make_sequences(df, target="power_kw", seq_len=36):

    feature_cols = [c for c in df.columns if c != target]

    X, y = [], []

    values = df.values

    target_idx = list(df.columns).index(target)

    feature_idx = [i for i, c in enumerate(df.columns) if c != target]

    for i in range(seq_len, len(values)):

        X.append(values[i - seq_len:i, feature_idx])
        y.append(values[i, target_idx])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"[SEQUENCES] {X.shape}")

    return X, y


# ============================================================
# MAIN PIPELINE
# ============================================================

if __name__ == "__main__":

    TARGET = "power_kw"
    SEQ_LEN = 36

    raw = load_raw("C:\\Users\\Faisal\\Downloads\\SCADA dataset\\T1.csv")

    featured = add_features(raw)

    train_df, val_df, test_df = split(featured)

    train_s, val_s, test_s = fit_and_scale(train_df, val_df, test_df)

    # LSTM datasets
    X_train, y_train = make_sequences(train_s, TARGET, SEQ_LEN)
    X_val, y_val = make_sequences(val_s, TARGET, SEQ_LEN)
    X_test, y_test = make_sequences(test_s, TARGET, SEQ_LEN)

    np.save(os.path.join(OUTPUT_DIR, "X_train_seq.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train_seq.npy"), y_train)

    np.save(os.path.join(OUTPUT_DIR, "X_val_seq.npy"), X_val)
    np.save(os.path.join(OUTPUT_DIR, "y_val_seq.npy"), y_val)

    np.save(os.path.join(OUTPUT_DIR, "X_test_seq.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test_seq.npy"), y_test)

    # Save ARIMA datasets
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"))
    val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"))
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"))

    print("\nProcessing complete")