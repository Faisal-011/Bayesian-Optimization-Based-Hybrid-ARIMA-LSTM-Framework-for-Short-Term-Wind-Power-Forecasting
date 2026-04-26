"""
Analysis & Visualization for Hybrid ARIMA-LSTM Wind Power Forecasting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading processed datasets...")

train_df = pd.read_csv("C:\\Users\\Faisal\\Downloads\\wind_forecasting\\processed\\train.csv", index_col=0)
test_df  = pd.read_csv("C:\\Users\\Faisal\\Downloads\\wind_forecasting\\processed\\test.csv", index_col=0)

train_series = train_df["power_kw"]
test_series  = test_df["power_kw"]

X_test_seq = np.load("C:\\Users\\Faisal\\Downloads\\wind_forecasting\\processed\\X_test_seq.npy")
y_test_seq = np.load("C:\\Users\\Faisal\\Downloads\\wind_forecasting\\processed\\y_test_seq.npy")


# ============================================================
# 2. LOAD TRAINED LSTM MODEL
# ============================================================

print("Loading trained LSTM model...")

lstm_model = load_model("lstm_hybrid_model.h5", compile=False)


# ============================================================
# 3. REBUILD ARIMA MODEL
# ============================================================

print("Rebuilding ARIMA model...")

arima_model = ARIMA(train_series, order=(3,1,2))
arima_fit = arima_model.fit()

arima_pred = arima_fit.forecast(steps=len(test_series))


# ============================================================
# 4. LSTM RESIDUAL PREDICTION
# ============================================================

print("Predicting nonlinear component with LSTM...")

lstm_residual_pred = lstm_model.predict(X_test_seq).flatten()


# ============================================================
# 5. HYBRID FORECAST
# ============================================================

print("Combining ARIMA + LSTM predictions...")

final_pred = arima_pred.values[:len(lstm_residual_pred)] + lstm_residual_pred

actual = test_series.values[:len(final_pred)]


# ============================================================
# 6. PERFORMANCE METRICS
# ============================================================

mae = mean_absolute_error(actual, final_pred)
rmse = np.sqrt(mean_squared_error(actual, final_pred))

print("\nHybrid Model Performance")
print("------------------------")
print("MAE :", mae)
print("RMSE:", rmse)


# ============================================================
# 7. FORECAST PLOT
# ============================================================

plt.figure(figsize=(12,6))

plt.plot(actual[:500], label="Actual Power")
plt.plot(final_pred[:500], label="Hybrid Forecast")

plt.title("Hybrid ARIMA–LSTM Wind Power Forecast")
plt.xlabel("Time Step")
plt.ylabel("Power (kW)")
plt.legend()

plt.show()


# ============================================================
# 8. ERROR DISTRIBUTION
# ============================================================

errors = actual - final_pred

plt.figure(figsize=(8,5))

plt.hist(errors, bins=50)
plt.title("Prediction Error Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")

plt.show()


# ============================================================
# 9. RESIDUAL PLOT
# ============================================================

plt.figure(figsize=(8,5))

plt.scatter(final_pred, errors, alpha=0.4)

plt.axhline(0, color="red", linestyle="--")

plt.xlabel("Predicted Power")
plt.ylabel("Residual Error")
plt.title("Residual Plot (Hybrid Model)")

plt.show()