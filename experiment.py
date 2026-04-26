# Train hybrid with default hyperparameters (no optimization)
# Use the same architecture but with default settings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from Model_training import X_train, y_train, X_val, y_val, X_test, actual, inverse_transform_power

lstm_default = Sequential([
    LSTM(64, return_sequences=True,
         input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dense(1)
])
lstm_default.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
lstm_default.fit(X_train, y_train,
                 validation_data=(X_val, y_val),
                 epochs=20, batch_size=64, verbose=0)

# Generate hybrid predictions with default LSTM
lstm_default_pred = lstm_default.predict(X_test).flatten()
lstm_default_kw = inverse_transform_power(lstm_default_pred)
hybrid_default = arima_test_aligned[:n] + lstm_default_kw[:n]

mae_default  = mean_absolute_error(actual, hybrid_default)
rmse_default = np.sqrt(mean_squared_error(actual, hybrid_default))
print(f"Default hybrid MAE: {mae_default:.2f} kW")
print(f"Default hybrid RMSE: {rmse_default:.2f} kW")