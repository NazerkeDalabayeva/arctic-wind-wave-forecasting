import xarray as xr
import pandas as pd

# Weather dataset
weather = xr.open_dataset(
    r"C:\Users\NDalabayeva\OneDrive - SLB\Desktop\For Academic paper\data\raw\data_stream-oper_stepType-instant.nc"
)

# Wave dataset
wave = xr.open_dataset(
    r"C:\Users\NDalabayeva\OneDrive - SLB\Desktop\For Academic paper\data\raw\data_stream-wave_stepType-instant.nc"
)

# Merge
merged = xr.merge([weather, wave])

print(merged)

print("\nVariables:")
print(list(merged.data_vars))

# Select one point near Svalbard
point = merged.sel(
    latitude=78.0,
    longitude=10.0,
    method="nearest"
)
print(point)
# Convert to DataFrame
df = point.to_dataframe().reset_index()
print(df.head())

import numpy as np
df["wind_speed"] = np.sqrt(
    df["u10"]**2 +
    df["v10"]**2
)
print(df[["valid_time", "wind_speed"]].head())

#Plot wind speed
import matplotlib.pyplot as plt
plt.figure(figsize=(12,5))
plt.plot(
df["valid_time"],
df["wind_speed"]
)
plt.title("Wind Speed at Svalbard")
plt.xlabel("Date")
plt.ylabel("Wind Speed (m/s)")
plt.grid(True)
plt.show()

print(df.isna().sum())

#Save as CSV
df.to_csv(
    r"C:\Users\NDalabayeva\OneDrive - SLB\Desktop\For Academic paper\data\processed\svalbard_data.csv",
    index=False
)
print("CSV saved successfully!")

#Correlation analysis
corr_cols = [
    "wind_speed",
    "t2m",
    "msl",
    "sst",
    "siconc",
    "swh",
    "mwp"
]
print(df[corr_cols].corr())

#Forecast target
df["target_24h"] = df["wind_speed"].shift(-24)
print(df[
    ["valid_time", "wind_speed", "target_24h"]
].head())

df = df.dropna()
print(df.shape)