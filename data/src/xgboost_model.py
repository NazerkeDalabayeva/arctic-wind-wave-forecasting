import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv(
    r"C:\Users\NDalabayeva\OneDrive - SLB\Desktop\For Academic paper\data\processed\svalbard_data.csv"
)

# Create target
df["target_24h"] = df["wind_speed"].shift(-24)

# Remove rows with missing target
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

# Time-series split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# XGBoost model
model = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
preds = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, preds)

print("XGBoost MAE:", mae)