import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv(r"C:\Users\Faisal\Downloads\wind_forecasting\processed\train.csv", index_col=0)

wind_speed = df['wind_speed']
power = df['power_kw']

# ── Fig 1: Wind Speed vs Power ──────────────────────────────────────────
plt.figure(figsize=(7, 5))
plt.style.use('default')
plt.scatter(wind_speed, power, s=5, alpha=0.3, color='steelblue')
plt.xlabel('Wind Speed (m/s)', fontsize=12)
plt.ylabel('Power Output (kW)', fontsize=12)
plt.title('Wind Speed vs Power Output', fontsize=13)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(r"C:\Users\Faisal\Downloads\wind_forecasting\fig1.png", 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("Fig 1 saved")

# ── Fig 2: Correlation Matrix ────────────────────────────────────────────
corr_matrix = df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, 
            cmap='coolwarm',
            center=0,
            vmin=-1, vmax=1,
            xticklabels=True, 
            yticklabels=True,
            annot=False,
            cbar_kws={'label': 'Pearson Correlation Coefficient'})
plt.xticks(fontsize=7, rotation=90)
plt.yticks(fontsize=7, rotation=0)
plt.title('Feature Correlation Matrix', fontsize=12)
plt.tight_layout()
plt.savefig(r"C:\Users\Faisal\Downloads\wind_forecasting\fig2.png", 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("Fig 2 saved")


# ── Fig 2: Scatter Matrix ────────────────────────────────────────────────

from pandas.plotting import scatter_matrix

scatter_cols = ['power_kw', 'wind_speed', 'theoretical_power', 'wind_dir']
subset = df[scatter_cols].sample(n=5000, random_state=42)  # sample for speed

rename_map = {
    'power_kw':          'LV ActivePower (kW)',
    'wind_speed':        'Wind Speed (m/s)',
    'theoretical_power': 'Theoretical_Power_Curve (KWh)',
    'wind_dir':          'Wind Direction (°)'
}
subset = subset.rename(columns=rename_map)

fig, axes = plt.subplots(4, 4, figsize=(10, 9))
scatter_matrix(subset, ax=axes, alpha=0.2, s=1,
               color='steelblue', diagonal='hist',
               hist_kwds={'color': 'steelblue', 'bins': 30})

for ax in axes.flatten():
    ax.tick_params(labelsize=6)
    xlabel = ax.get_xlabel()
    ylabel = ax.get_ylabel()
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=7)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7)

plt.suptitle('Scatter Matrix of Key SCADA Features', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(r"C:\Users\Faisal\Downloads\wind_forecasting\fig_scatter_matrix.png",
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("Scatter matrix saved")


# ── Fig 3: Training and Validation Loss + MAE ────────────────────────────
# You need to save history during LSTM training first.
# Add this to your LSTM training script:
#
#   history = lstm_model.fit(X_train, y_train,
#                            validation_data=(X_val, y_val),
#                            epochs=20, batch_size=32)
#   import pickle
#   with open(r"C:\Users\Faisal\Downloads\wind_forecasting\history.pkl", "wb") as f:
#       pickle.dump(history.history, f)

import pickle

with open(r"C:\Users\Faisal\Downloads\wind_forecasting\history.pkl", "rb") as f:
    hist = pickle.load(f)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss
axes[0].plot(hist['loss'],     label='Training Loss',   color='blue')
axes[0].plot(hist['val_loss'], label='Validation Loss', color='orange')
axes[0].set_title('Training and Validation Loss', fontsize=11)
axes[0].set_xlabel('Epochs', fontsize=10)
axes[0].set_ylabel('Loss', fontsize=10)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# MAE
axes[1].plot(hist['mae'],     label='Training MAE',   color='red')
axes[1].plot(hist['val_mae'], label='Validation MAE', color='green')
axes[1].set_title('Training and Validation MAE', fontsize=11)
axes[1].set_xlabel('Epochs', fontsize=10)
axes[1].set_ylabel('MAE', fontsize=10)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r"C:\Users\Faisal\Downloads\wind_forecasting\fig_training_curves.png",
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("Training curves saved")