# Bayesian Optimization-Based Hybrid ARIMA-LSTM Framework for Short-Term Wind Power Forecasting

This repository contains the full implementation for the research paper: **"Bayesian Optimization-Based Hybrid ARIMA–LSTM Framework for Short-Term Wind Power Forecasting"** by Mohammed Faisal and Sridhar S., submitted to the *Journal of Renewable and Sustainable Energy (JRSE)*.

## Overview

A hybrid ARIMA+LSTM forecasting framework for 10-minute-ahead wind power prediction using real turbine SCADA data. The framework decomposes the wind power signal into a linear component (ARIMA) and a nonlinear residual (LSTM), with Bayesian hyperparameter optimization via Optuna applied over the full hybrid validation error.

**Key results on held-out test set:**
- MAE: 172.13 kW (86.4% reduction over ARIMA baseline)
- RMSE: 270.20 kW (84.1% reduction over ARIMA baseline)
- Outperforms standalone LSTM, XGBoost, and Transformer encoder on RMSE

## Dataset

Wind turbine SCADA data (2018, 10-minute resolution, ~50,530 records) from:
> B. Erisen, "Wind turbine SCADA dataset," Kaggle, 2019.
> https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset

Download and place in a `data/` folder before running.

## Repository Structure

```
├── data_processing.py          # Data cleaning, feature engineering (29 features)
├── data_processing_v2.py       # Updated preprocessing pipeline
├── Model_training.py           # ARIMA fitting + LSTM training
├── modeltraining2.py           # Updated training script (final version)
├── baseline.py                 # Standalone LSTM, XGBoost, Transformer baselines
├── experiment.py               # Full hybrid experiment pipeline
├── analysis.py                 # Results analysis and metrics
├── data_vis.py                 # Figures and visualizations
├── processed/                  # Intermediate processed files (generated at runtime)
├── best_params.pkl             # Optuna best hyperparameters (v1)
├── best_params_v2.pkl          # Optuna best hyperparameters (v2, final)
├── lstm_hybrid_model.h5        # Trained LSTM model (v1)
├── lstm_hybrid_model_fixed.keras   # Trained LSTM model (fixed)
├── lstm_hybrid_model_v2.keras  # Trained LSTM model (final)
├── baseline_comparison_results.csv     # Baseline model metrics
├── baseline_comparison_v2_results.csv  # Final baseline metrics
└── figures/                    # Output figures (PNG)
```

## Requirements

```
Python 3.9+
numpy
pandas
scikit-learn
statsmodels
tensorflow >= 2.10
optuna
xgboost
matplotlib
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

**1. Preprocess data:**
```bash
python data_processing_v2.py
```

**2. Train hybrid model:**
```bash
python modeltraining2.py
```

**3. Run baselines:**
```bash
python baseline.py
```

**4. Full experiment:**
```bash
python experiment.py
```

**5. Generate figures:**
```bash
python data_vis.py
```

## Methodology

- **Feature Engineering:** 4 raw SCADA signals → 29 features including wind direction decomposition, cyclic time encoding, lag variables, rolling statistics, and novel power ratio index
- **ARIMA:** Order (3,1,2) fitted on training data; generates linear forecasts and residuals
- **LSTM:** Two stacked layers (43 + 21 units), trained on ARIMA residuals
- **Bayesian Optimization:** 30 trials via Optuna, objective defined over full hybrid validation RMSE
- **Hybrid output:** ARIMA forecast + LSTM residual prediction

## Results

| Model | MAE (kW) | RMSE (kW) |
|---|---|---|
| ARIMA (3,1,2) | 1266 | 1695 |
| Standalone LSTM | 180 | 277 |
| XGBoost | 172 | 276 |
| Transformer (encoder-only) | 283 | 412 |
| **Hybrid ARIMA-LSTM (Proposed)** | **172** | **270** |

## Citation

If you use this code, please cite the paper (citation details to be updated upon publication).

## License

MIT License
