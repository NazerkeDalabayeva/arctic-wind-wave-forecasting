import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# LOAD DATA
# ==========================================

weather = xr.open_dataset(
    r"C:\Users\NDalabayeva\OneDrive - SLB\Desktop\For Academic paper\data\raw\data_stream-oper_stepType-instant.nc"
)

wave = xr.open_dataset(
    r"C:\Users\NDalabayeva\OneDrive - SLB\Desktop\For Academic paper\data\raw\data_stream-wave_stepType-instant.nc"
)

print("\nWeather Variables:")
print(list(weather.data_vars))

print("\nWave Variables:")
print(list(wave.data_vars))

# ==========================================
# SELECT OCEAN POINT NEAR SVALBARD
# ==========================================

latitude = 78.0
longitude = 10.0

weather_point = weather.sel(
    latitude=latitude,
    longitude=longitude,
    method="nearest"
)

wave_point = wave.sel(
    latitude=latitude,
    longitude=longitude,
    method="nearest"
)

# ==========================================
# CONVERT TO DATAFRAMES
# ==========================================

weather_df = weather_point.to_dataframe().reset_index()

wave_df = wave_point.to_dataframe().reset_index()

# ==========================================
# CALCULATE WIND SPEED
# ==========================================

weather_df["wind_speed"] = np.sqrt(
    weather_df["u10"]**2 +
    weather_df["v10"]**2
)

print("\nWind Speed Sample:")
print(
    weather_df[
        ["valid_time", "wind_speed"]
    ].head()
)

# ==========================================
# WIND SPEED FIGURE
# ==========================================

plt.figure(figsize=(12, 5))

plt.plot(
    weather_df["valid_time"],
    weather_df["wind_speed"]
)

plt.title("Wind Speed at Svalbard")
plt.xlabel("Date")
plt.ylabel("Wind Speed (m/s)")
plt.grid(True)

plt.savefig(
    r"C:\Users\NDalabayeva\OneDrive - SLB\Desktop\For Academic paper\figures\wind_speed_svalbard.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================================
# WAVE HEIGHT FIGURE
# ==========================================

plt.figure(figsize=(12, 5))

plt.plot(
    wave_df["valid_time"],
    wave_df["swh"]
)

plt.title("Significant Wave Height at Svalbard")
plt.xlabel("Date")
plt.ylabel("Wave Height (m)")
plt.grid(True)

plt.savefig(
    r"C:\Users\NDalabayeva\OneDrive - SLB\Desktop\For Academic paper\figures\wave_height_svalbard.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================================
# MERGE DATA
# ==========================================

df = pd.merge(
    weather_df,
    wave_df[
        [
            "valid_time",
            "mwd",
            "mwp",
            "swh"
        ]
    ],
    on="valid_time"
)

print("\nMerged Data:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nMissing Values:")
print(df.isna().sum())