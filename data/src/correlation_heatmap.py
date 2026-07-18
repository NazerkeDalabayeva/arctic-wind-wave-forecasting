import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(
    r"C:\Users\NDalabayeva\OneDrive - SLB\Desktop\For Academic paper\data\processed\svalbard_data.csv"
)

df["wind_speed"] = (
    df["u10"]**2 +
    df["v10"]**2
) ** 0.5

cols = [
    "wind_speed",
    "t2m",
    "msl",
    "sst",
    "swh",
    "mwp"
]

plt.figure(figsize=(8,6))

sns.heatmap(
    df[cols].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix")

plt.tight_layout()

plt.savefig(
    r"C:\Users\NDalabayeva\OneDrive - SLB\Desktop\For Academic paper\figures\correlation_heatmap.png",
    dpi=300
)

plt.show()