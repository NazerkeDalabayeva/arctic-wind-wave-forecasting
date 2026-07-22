import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# LOAD WEATHER DATASETS
# ============================================================

weather_1 = xr.open_dataset(
    r"C:\Users\NDalabayeva\OneDrive - SLB\Desktop\For Academic paper\data\raw\data_stream-oper_stepType-instant.nc"
)

weather_2 = xr.open_dataset(
    r"C:\Users\NDalabayeva\OneDrive - SLB\Desktop\For Academic paper\data\raw\April-Jun\data_stream-oper_stepType-instant_Apr-May-June.nc"
)

weather = xr.concat(
    [weather_1, weather_2],
    dim="valid_time"
)

# ============================================================
# LOAD WAVE DATASETS
# ============================================================

wave_1 = xr.open_dataset(
    r"C:\Users\NDalabayeva\OneDrive - SLB\Desktop\For Academic paper\data\raw\data_stream-wave_stepType-instant.nc"
)

wave_2 = xr.open_dataset(
    r"C:\Users\NDalabayeva\OneDrive - SLB\Desktop\For Academic paper\data\raw\April-Jun\data_stream-wave_stepType-instant_Apr-May-June.nc"
)

wave = xr.concat(
    [wave_1, wave_2],
    dim="valid_time"
)

# ============================================================
# MERGE WEATHER + WAVE
# ============================================================

merged = xr.merge([weather, wave])

print("\nVariables:")
print(list(merged.data_vars))

# ============================================================
# SELECT REPRESENTATIVE MIZ LOCATION
# ============================================================

latitude = 76.0
longitude = 15.0

point = merged.sel(
    latitude=latitude,
    longitude=longitude,
    method="nearest"
)

print("\nSelected Arctic MIZ Location")
print(f"Latitude : {latitude}")
print(f"Longitude: {longitude}")

# ============================================================
# CONVERT TO DATAFRAME
# ============================================================

df = point.to_dataframe().reset_index()

# ============================================================
# FEATURE ENGINEERING
# ============================================================

df["wind_speed"] = np.sqrt(
    df["u10"]**2 +
    df["v10"]**2
)

# ============================================================
# SEA ICE FORECAST TARGET
# ============================================================

df["target_24h"] = df["siconc"].shift(-24)

# Remove final rows with missing target
df = df.dropna()

# ============================================================
# DATASET SUMMARY
# ============================================================

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isna().sum())

print("\nSea Ice Statistics:")
print(df["siconc"].describe())

# ============================================================
# CORRELATION ANALYSIS
# ============================================================

corr_cols = [
    "siconc",
    "wind_speed",
    "t2m",
    "msl",
    "sst",
    "swh",
    "mwp",
    "mwd"
]

print("\nCorrelation Matrix:")
print(df[corr_cols].corr())

# ============================================================
# VISUALIZATION
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    df["valid_time"],
    df["wind_speed"]
)

plt.title("Arctic Wind Speed at MIZ Location")
plt.xlabel("Date")
plt.ylabel("Wind Speed (m/s)")
plt.grid(True)

plt.savefig(
    r"C:\Users\NDalabayeva\OneDrive - SLB\Desktop\For Academic paper\figures\arctic_wind_speed_timeseries.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# SAVE FINAL MACHINE LEARNING DATASET
# ============================================================

output_path = (
    r"C:\Users\NDalabayeva\OneDrive - SLB\Desktop\For Academic paper\data\processed\arctic_sea_ice_forecasting_dataset.csv"
)

df.to_csv(
    output_path,
    index=False
)

print("\nCSV saved successfully!")

print(f"\nOutput file:\n{output_path}")

print(
    "\nDataset prepared for Arctic sea-ice forecasting and decision-support modelling."
)