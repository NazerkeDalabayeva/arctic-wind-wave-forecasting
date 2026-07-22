import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ============================================================
# LOAD PROCESSED DATASET
# ============================================================

df = pd.read_csv(
    r"C:\Users\NDalabayeva\OneDrive - SLB\Desktop\For Academic paper\data\processed\arctic_sea_ice_forecasting_dataset.csv"
)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

# ============================================================
# TARGET STATISTICS
# ============================================================

print("\nTarget Statistics:")
print(df["target_24h"].describe())

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
# TIME-SERIES TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

print("\nTraining Shape:")
print(X_train.shape)

print("\nTesting Shape:")
print(X_test.shape)

# ============================================================
# TRAIN MODEL
# ============================================================

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

# ============================================================
# PREDICTIONS
# ============================================================

preds = model.predict(X_test)

# ============================================================
# EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    preds
)

print("\n===================================")
print("LINEAR REGRESSION RESULTS")
print("===================================")

print(f"MAE: {mae:.6f}")

# ============================================================
# FEATURE COEFFICIENTS
# ============================================================

coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})

print("\nFeature Coefficients:")

print(
    coefficients.sort_values(
        by="Coefficient",
        ascending=False
    )
)

# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

comparison = pd.DataFrame({
    "Actual": y_test.values[:10],
    "Predicted": preds[:10]
})

print("\nFirst 10 Predictions:")

print(comparison)