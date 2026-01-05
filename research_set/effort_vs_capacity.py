
# === Effort vs Capacity (Tonnage vs Fishing Hours) ===
import pandas as pd, matplotlib.pyplot as plt, numpy as np
from pathlib import Path

BASE = Path("/Users/harinisaravanan/Documents/Python_FishingVessels/Processed/analysis")
tonnage = pd.read_parquet(BASE / "derived/annual_tonnage_by_country.parquet")
summary = pd.read_parquet(BASE / "static_summary_country_year.parquet")

df = tonnage.merge(summary, on=["flag","year"], how="inner")
df = df[df["tonnage"] > 0]
df["efficiency"] = df["total_fishing_hours"] / df["tonnage"]

# Filter for top 10 countries to make plots manageable
exclude = {"CHN", "UNKNOWN-CHN"}
top10 = (
    summary[~summary["flag"].isin(exclude)]
    .groupby("flag")["total_fishing_hours"].sum()
    .nlargest(10).index.tolist()
)
df = df[df["flag"].isin(top10)]


# --- Create a separate plot for each country ---
norm = plt.Normalize(df["year"].min(), df["year"].max())
cmap = plt.get_cmap("viridis")

for flag in top10:
    fig, ax = plt.subplots(figsize=(8, 6)) # Create a new figure and axes for each plot
    
    country_df = df[df["flag"] == flag]

    # Create scatter plot for the country
    scatter = ax.scatter(
        country_df["tonnage"] / 1e6,
        country_df["total_fishing_hours"] / 1e6,
        c=country_df["year"],
        cmap=cmap,
        norm=norm,
        alpha=0.7,
        edgecolors="k",
        linewidth=0.5
    )
    
    # Add titles and labels for each plot
    ax.set_xlabel(f"Total Fleet Tonnage (Million GT), {flag}", fontsize=14, fontweight='bold')
    ax.set_ylabel("Total Fishing Hours (Million Hours)", fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Add a colorbar for each plot
    cb = fig.colorbar(scatter, ax=ax)
    cb.set_label("Year")
    
    # Adjust layout and show the plot
    fig.tight_layout()
    plt.show()
