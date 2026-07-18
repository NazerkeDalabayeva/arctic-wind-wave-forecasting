import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv(
    r"C:\Users\NDalabayeva\OneDrive - SLB\Desktop\For Academic paper\data\processed\svalbard_data.csv"
)

# Forecast target
df["target_24h"] = df["wind_speed"].shift(-24)

# Remove final 24 rows
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
    shuffle=False,
    test_size=0.2
)

# Train
model = LinearRegression()

model.fit(
    X_train,
    y_train
)

# Predict
preds = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(
    y_test,
    preds
)
print("Linear Regression MAE:", mae)