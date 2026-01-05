
# === COVID Analysis (Annual) — Top-10 Countries (Excluding China) ===
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Rectangle

BASE = Path("/Users/harinisaravanan/Documents/Python_FishingVessels/Processed")
ANALYSIS = BASE / "analysis"
SUMMARY = ANALYSIS / "static_summary_country_year.parquet"

df = pd.read_parquet(SUMMARY).copy()
df["year"] = df["year"].astype(int)
df = df.dropna(subset=["flag", "year", "total_fishing_hours"])

exclude = {"CHN", "UNKNOWN-CHN"}
top10 = (df[~df["flag"].isin(exclude)]
         .groupby("flag", as_index=False)["total_fishing_hours"].sum()
         .sort_values("total_fishing_hours", ascending=False)
         .head(10)["flag"].tolist())

sub = df[df["flag"].isin(top10)].copy()

# Period classifier (annual)
def period(y):
    if 2012 <= y <= 2019: return "Before"
    if 2020 <= y <= 2021: return "During"
    if 2022 <= y <= 2024: return "After"
    return "Other"

sub["period"] = sub["year"].apply(period)

# Pivot for line plot
piv = sub.pivot_table(index="year", columns="flag", values="total_fishing_hours",
                      aggfunc="sum").sort_index()

fig, ax = plt.subplots(figsize=(14, 7))
cmap = plt.get_cmap("tab20")
colors = {f: cmap(i % 20) for i, f in enumerate(top10)}

# Shade periods (Before, During, After)
years = list(piv.index)
if len(years):
    y_min, y_max = min(years), max(years)
    # Before: 2012-2019
    if y_min <= 2019:
        x0 = max(y_min, 2012); x1 = min(y_max, 2019)
        if x0 <= x1: ax.axvspan(x0-0.5, x1+0.5, color="lightgray", alpha=0.35, zorder=0)
    # During: 2020-2021
    if (y_min <= 2021) and (y_max >= 2020):
        ax.axvspan(2020-0.5, 2021+0.5, color="khaki", alpha=0.35, zorder=0)
    # After: 2022-2024
    if y_max >= 2022:
        ax.axvspan(2022-0.5, min(y_max, 2024)+0.5, color="honeydew", alpha=0.35, zorder=0)

# Redraw lines on top
for f in top10:
    if f in piv.columns:
        ax.plot(piv.index, piv[f]/1e6, lw=2, label=f, color=colors[f])

ax.set_ylabel("Fishing Hours (Millions)", fontsize=14, fontweight='bold')
ax.tick_params(axis='x', labelsize=12)
ax.grid(alpha=0.25)
ax.legend(ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig(ANALYSIS / "figs" / "covid_annual_top10_excl_china.png", dpi=200)
plt.show()