import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    r"C:\Users\NDalabayeva\OneDrive - SLB\Desktop\For Academic paper\data\processed\arctic_sea_ice_forecasting_dataset.csv"
)

print("\nDataset Shape:")
print(df.shape)

print("\nMaximum Ice Concentration:")
print(df["target_24h"].max())


# ============================================================
# FEATURES / TARGET
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
# LINEAR REGRESSION
# ============================================================

lr = LinearRegression()

lr.fit(X_train, y_train)

lr_preds = lr.predict(X_test)

lr_preds = lr_preds.clip(0, 1)


# ============================================================
# XGBOOST
# ============================================================

xgb = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)

xgb.fit(X_train, y_train)

xgb_preds = xgb.predict(X_test)

xgb_preds = xgb_preds.clip(0, 1)


# ============================================================
# METRICS
# ============================================================

lr_mae = mean_absolute_error(y_test, lr_preds)
lr_rmse = mean_squared_error(y_test, lr_preds) ** 0.5
lr_r2 = r2_score(y_test, lr_preds)

xgb_mae = mean_absolute_error(y_test, xgb_preds)
xgb_rmse = mean_squared_error(y_test, xgb_preds) ** 0.5
xgb_r2 = r2_score(y_test, xgb_preds)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("\nLinear Regression")
print(f"MAE  = {lr_mae:.6f}")
print(f"RMSE = {lr_rmse:.6f}")
print(f"R²   = {lr_r2:.6f}")

print("\nXGBoost")
print(f"MAE  = {xgb_mae:.6f}")
print(f"RMSE = {xgb_rmse:.6f}")
print(f"R²   = {xgb_r2:.6f}")


# ============================================================
# FIND ICE EVENTS IN WHOLE DATASET
# ============================================================

ice_events = df[df["target_24h"] > 0]

print("\nNumber of Ice Event Records:")
print(len(ice_events))

if len(ice_events) == 0:
    print("\nNo ice events found.")
else:

    # First meaningful ice event

    first_ice_index = ice_events.index[0]

    start_idx = max(first_ice_index - 150, 0)
    end_idx = min(first_ice_index + 150, len(df))

    dates = pd.to_datetime(
        df.loc[start_idx:end_idx, "valid_time"]
    )

    actual = df.loc[start_idx:end_idx, "target_24h"]

    # Rebuild predictions on this window

    full_lr_preds = lr.predict(X).clip(0, 1)
    full_xgb_preds = xgb.predict(X).clip(0, 1)

    lr_window = full_lr_preds[start_idx:end_idx + 1]
    xgb_window = full_xgb_preds[start_idx:end_idx + 1]

    # ========================================================
    # PLOT ICE EVENT
    # ========================================================

    plt.figure(figsize=(15, 6))

    plt.plot(
        dates,
        actual,
        color="black",
        linewidth=3,
        label="Observed Sea Ice Concentration"
    )

    plt.plot(
        dates,
        lr_window,
        "--",
        linewidth=2,
        label="Linear Regression"
    )

    plt.plot(
        dates,
        xgb_window,
        ":",
        linewidth=3,
        label="XGBoost"
    )

    plt.title(
        "Sea-Ice Forecasting During Arctic Ice Event"
    )

    plt.xlabel("Date")
    plt.ylabel("Sea Ice Concentration")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        r"C:\Users\NDalabayeva\OneDrive - SLB\Desktop\For Academic paper\figures\sea_ice_model_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        "\nFigure saved: sea_ice_model_comparison.png"
    )