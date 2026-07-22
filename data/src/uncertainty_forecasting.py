import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    r"C:\Users\NDalabayeva\OneDrive - SLB\Desktop\For Academic paper\data\processed\arctic_sea_ice_forecasting_dataset.csv"
)

# ============================================================
# FEATURES AND TARGET
# ============================================================

features = [
    "siconc",
    "wind_speed",
    "t2m",
    "msl",
    "sst",
    "swh",
    "mwp",
    "mwd"
]

X = df[features]
y = df["target_24h"]

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# ============================================================
# BOOTSTRAP ENSEMBLE
# ============================================================

predictions = []

for i in range(100):

    sample_idx = np.random.choice(
        len(X_train),
        len(X_train),
        replace=True
    )

    X_boot = X_train.iloc[sample_idx]
    y_boot = y_train.iloc[sample_idx]

    model = DecisionTreeRegressor(
        max_depth=5,
        random_state=i
    )

    model.fit(X_boot, y_boot)

    pred = model.predict(X)

    predictions.append(pred)

predictions = np.array(predictions)

# ============================================================
# FIND LARGEST SEA-ICE EVENT
# ============================================================

max_idx = df["target_24h"].idxmax()

window = 150

start_idx = max(max_idx - window, 0)
end_idx = min(max_idx + window, len(df))

plot_df = df.iloc[start_idx:end_idx]

plot_dates = pd.to_datetime(
    plot_df["valid_time"]
)

plot_actual = plot_df["target_24h"].values

# ============================================================
# UNCERTAINTY ESTIMATES FOR EVENT WINDOW
# ============================================================

event_predictions = predictions[:, start_idx:end_idx]

mean_pred = event_predictions.mean(axis=0)

lower_bound = np.percentile(
    event_predictions,
    2.5,
    axis=0
)

upper_bound = np.percentile(
    event_predictions,
    97.5,
    axis=0
)

# Physical limits

mean_pred = np.clip(mean_pred, 0, 1)
lower_bound = np.clip(lower_bound, 0, 1)
upper_bound = np.clip(upper_bound, 0, 1)

# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(15, 6))

plt.plot(
    plot_dates,
    plot_actual,
    color="black",
    linewidth=3,
    label="Observed Sea Ice Concentration"
)

plt.plot(
    plot_dates,
    mean_pred,
    color="blue",
    linewidth=2,
    label="Ensemble Mean Forecast"
)

plt.fill_between(
    plot_dates,
    lower_bound,
    upper_bound,
    color="lightblue",
    alpha=0.4,
    label="95% Prediction Interval"
)

plt.title(
    "Sea-Ice Forecast Uncertainty During Major Ice Event"
)

plt.xlabel("Date")
plt.ylabel("Sea Ice Concentration")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    r"C:\Users\NDalabayeva\OneDrive - SLB\Desktop\For Academic paper\figures\forecast_uncertainty.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# SUMMARY
# ============================================================

print("\nForecast uncertainty analysis completed.")

print("\nMaximum observed sea ice concentration:")
print(df['target_24h'].max())

print("\nIce event plotted around index:")
print(max_idx)