import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# LOAD DATA
# ==========================================

# Jan-Mar weather
weather_1 = xr.open_dataset(
    r"C:\Users\NDalabayeva\OneDrive - SLB\Desktop\For Academic paper\data\raw\data_stream-oper_stepType-instant.nc"
)

# Apr-Jun weather
weather_2 = xr.open_dataset(
    r"C:\Users\NDalabayeva\OneDrive - SLB\Desktop\For Academic paper\data\raw\April-Jun\data_stream-oper_stepType-instant_Apr-May-June.nc"
)

# Combine in time
weather = xr.concat(
    [weather_1, weather_2],
    dim="valid_time"
)

# Jan-Mar wave
wave_1 = xr.open_dataset(
    r"C:\Users\NDalabayeva\OneDrive - SLB\Desktop\For Academic paper\data\raw\data_stream-wave_stepType-instant.nc"
)

# Apr-Jun wave
wave_2 = xr.open_dataset(
    r"C:\Users\NDalabayeva\OneDrive - SLB\Desktop\For Academic paper\data\raw\April-Jun\data_stream-wave_stepType-instant_Apr-May-June.nc"
)

# Combine in time
wave = xr.concat(
    [wave_1, wave_2],
    dim="valid_time"
)

print("\nWeather Variables:")
print(list(weather.data_vars))

print("\nWave Variables:")
print(list(wave.data_vars))

print(weather.dims)
print(wave.dims)

# ==========================================
# SELECT REPRESENTATIVE MARGINAL ICE ZONE
# LOCATION NEAR SVALBARD
# ==========================================

latitude = 76.0
longitude = 15.0

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
print("\nSea Ice Statistics:")
print(
    weather_df["siconc"].describe()
)

# ==========================================
# WIND SPEED FIGURE
# ==========================================

plt.figure(figsize=(12, 5))

plt.plot(
    weather_df["valid_time"],
    weather_df["wind_speed"]
)

plt.title("Wind Speed at MIZ Location Near Svalbard")
plt.xlabel("Date")
plt.ylabel("Wind Speed (m/s)")
plt.grid(True)

plt.savefig(
    r"C:\Users\NDalabayeva\OneDrive - SLB\Desktop\For Academic paper\figures\arctic_wind_speed_timeseries.png",
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

plt.title("Arctic Significant Wave Height at MIZ Location")
plt.xlabel("Date")
plt.ylabel("Wave Height (m)")
plt.grid(True)

plt.savefig(
    r"C:\Users\NDalabayeva\OneDrive - SLB\Desktop\For Academic paper\figures\arctic_wave_height_timeseries.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================================
# SEA ICE CONCENTRATION FIGURE
# ==========================================

plt.figure(figsize=(12, 5))

plt.plot(
    weather_df["valid_time"],
    weather_df["siconc"]
)

plt.title("Sea Ice Concentration at Arctic MIZ Location")
plt.xlabel("Date")
plt.ylabel("Sea Ice Concentration")
plt.grid(True)

plt.savefig(
    r"C:\Users\NDalabayeva\OneDrive - SLB\Desktop\For Academic paper\figures\sea_ice_concentration_timeseries.png",
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

print("\nMerged Arctic Environmental Dataset:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nMissing Values:")
print(df.isna().sum())

print(weather_point.latitude.values)
print(weather_point.longitude.values)

print(wave_point.latitude.values)
print(wave_point.longitude.values)