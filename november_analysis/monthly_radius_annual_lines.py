# === Monthly Fleet Radius by Year — Top-10 (Excluding China) ===
import math
from pathlib import Path
import pandas as pd, numpy as np, matplotlib.pyplot as plt

BASE = Path("/Users/harinisaravanan/Documents/Python_FishingVessels/Processed/analysis")
SUMMARY = BASE / "static_summary_country_year.parquet"    # to get top10
R_MONTH = BASE / "static_radius_monthly.parquet"          # flag, year, month, radius_km

summary_df = pd.read_parquet(SUMMARY)
r = pd.read_parquet(R_MONTH)

exclude = {"CHN", "UNKNOWN-CHN"}
top10 = (summary_df[~summary_df["flag"].isin(exclude)]
         .groupby("flag", as_index=False)["total_fishing_hours"].sum()
         .sort_values("total_fishing_hours", ascending=False)
         .head(10)["flag"].tolist())
print("Top-10 flags (excl China):", top10)

r = r[r["flag"].isin(top10)].copy()
r["year"] = r["year"].astype(int)
r["month"] = r["month"].astype(int)

# --- Create a separate plot for each country ---
cmap = plt.get_cmap("tab20")
years_sorted = sorted(r["year"].unique())
year_colors = {y: cmap(i % 20) for i, y in enumerate(years_sorted)}

for c in top10:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sub = r[r["flag"] == c].copy()
    for y in years_sorted:
        s = sub[sub["year"] == y].sort_values("month")
        if not s.empty:
            ax.plot(s["month"], s["radius_km"], lw=1.8, label=str(y), color=year_colors[y])

    ax.set_title(f"{c}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month", fontsize=14, fontweight='bold')
    ax.set_ylabel("Radius (km)", fontsize=14, fontweight='bold')

    ax.set_xlim(1, 12)
    ax.set_xticks(range(1, 13))
    ax.grid(alpha=0.2)

    ax.legend(title="Year", ncol=2, bbox_to_anchor=(1, 1), loc="upper left")
    
    plt.tight_layout(rect=[0, 0, 0.95, 1]) # Adjust layout to make room for legend
    plt.show()

# Create the output directory if it doesn't exist, as the original script did.
(out := BASE / "figs").mkdir(exist_ok=True, parents=True)
