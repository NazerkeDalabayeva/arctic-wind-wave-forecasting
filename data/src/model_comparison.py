import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv(
    r"C:\Users\NDalabayeva\OneDrive - SLB\Desktop\For Academic paper\data\processed\svalbard_data.csv"
)

# Create target
df["target_24h"] = df["wind_speed"].shift(-24)

# Remove missing rows
df = df.dropna()

# Features
features = [
    "wind_speed",
    "t2m",
    "msl",
    "sst",
    "swh",
    "mwp"
]

X = df[features]
y = df["target_24h"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

lr_preds = lr.predict(X_test)

# XGBoost
xgb = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)

xgb.fit(X_train, y_train)

xgb_preds = xgb.predict(X_test)

# Plot
dates = pd.to_datetime(df.iloc[-len(y_test):]["valid_time"])

plt.figure(figsize=(15,6))

plt.plot(dates[:200], y_test.values[:200], label="Actual")
plt.plot(dates[:200], lr_preds[:200], "--", label="Linear Regression")
plt.plot(dates[:200], xgb_preds[:200], ":", label="XGBoost")

plt.title("24-Hour Wind Speed Forecast at Svalbard")
plt.xlabel("Date")
plt.ylabel("Wind Speed (m/s)")

plt.xticks(
    dates[:200:24],   # every 24 hours
    rotation=45
)

plt.legend()
plt.grid(True)
plt.tight_layout()

# Save figure
plt.savefig(
    r"C:\Users\NDalabayeva\OneDrive - SLB\Desktop\For Academic paper\figures\model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()