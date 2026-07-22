import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# LOAD ARCTIC MIZ DATASET
# ==========================================

df = pd.read_csv(
    r"C:\Users\NDalabayeva\OneDrive - SLB\Desktop\For Academic paper\data\processed\arctic_sea_ice_forecasting_dataset.csv"
)

# ==========================================
# CALCULATE WIND SPEED
# ==========================================

df["wind_speed"] = (
    df["u10"]**2 +
    df["v10"]**2
) ** 0.5

# ==========================================
# VARIABLES FOR CORRELATION ANALYSIS
# ==========================================

cols = [
    "siconc",
    "wind_speed",
    "t2m",
    "msl",
    "sst",
    "swh",
    "mwp",
    "mwd"
]

# ==========================================
# CORRELATION MATRIX
# ==========================================

plt.figure(figsize=(10, 8))

sns.heatmap(
    df[cols].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title(
    "Correlation Matrix of Arctic Environmental Variables"
)

plt.tight_layout()

plt.savefig(
    r"C:\Users\NDalabayeva\OneDrive - SLB\Desktop\For Academic paper\figures\environmental_correlation_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()