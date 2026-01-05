'''
Question:

How has fleet efficiency (gross tonnage per fishing hour) evolved over time?

Interpretation Goal:
Quantify recovery of productivity post-pandemic — normalized by effort.

'''

# === Fleet Efficiency Trend (GT per Hour) ===
import pandas as pd, matplotlib.pyplot as plt, numpy as np
from pathlib import Path

BASE = Path("/Users/harinisaravanan/Documents/Python_FishingVessels/Processed/analysis")
tonnage = pd.read_parquet(BASE / "derived/annual_tonnage_by_country.parquet")
summary = pd.read_parquet(BASE / "static_summary_country_year.parquet")

df = tonnage.merge(summary, on=["flag","year"], how="inner")
df["GT_per_hour"] = df["tonnage"] / df["total_fishing_hours"]

exclude = {"CHN","UNKNOWN-CHN"}
top10 = (
    summary[~summary["flag"].isin(exclude)]
    .groupby("flag")["total_fishing_hours"].sum()
    .nlargest(10).index.tolist()
)
df = df[df["flag"].isin(top10)]

cmap = plt.get_cmap("tab20")
colors = {f: cmap(i % 20) for i, f in enumerate(top10)}

fig, ax = plt.subplots(figsize=(11,6))
for f in top10:
    sub = df[df["flag"]==f].sort_values("year")
    ax.plot(sub["year"], sub["GT_per_hour"], lw=2, label=f, color=colors[f])
ax.axvspan(2020, 2021, color="grey", alpha=0.2)
ax.set_ylabel("Gross Tonnage per Fishing Hour (GT/h)", fontsize=12, fontweight="bold")
ax.set_xlabel("Year", fontsize=12, fontweight="bold")
ax.grid(alpha=0.3)
ax.legend(ncol=2, bbox_to_anchor=(1.02,1), loc="upper left")
plt.tight_layout()
plt.show()


