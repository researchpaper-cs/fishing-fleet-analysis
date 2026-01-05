# === Annual Fishing Hours — Top-10 Countries (Excluding China) ===
import duckdb, pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE = Path("/Users/harinisaravanan/Documents/Python_FishingVessels/Processed")
ANALYSIS = BASE / "analysis"
STATIC_GLOB = str(BASE / "static" / "year=*" / "*.parquet")   # raw static parquet
SUMMARY = ANALYSIS / "static_summary_country_year.parquet"     # flag, year, total_fishing_hours

OUTDIR = ANALYSIS / "figs"
OUTDIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load summary (country-year totals)
# -----------------------------
df = pd.read_parquet(SUMMARY).copy()
df["year"] = df["year"].astype(int)
df = df.dropna(subset=["flag", "year", "total_fishing_hours"])

# -----------------------------
# Build Top-10 (excluding China)
# -----------------------------
exclude = {"CHN", "UNKNOWN-CHN"}
totals = (df[~df["flag"].isin(exclude)]
          .groupby("flag", as_index=False)["total_fishing_hours"].sum()
          .sort_values("total_fishing_hours", ascending=False))
top10_flags = totals.head(10)["flag"].tolist()
print("Top-10 flags (excl. China):", top10_flags)

# Filter to top10
sub = df[df["flag"].isin(top10_flags)].copy()

# -----------------------------
# Combined clustered bars (all years on x, countries as groups)
# -----------------------------
def plot_clustered_bars(data, title, savepath):
    years = sorted(data["year"].unique())
    flags = top10_flags

    piv = data.pivot_table(index="year", columns="flag", values="total_fishing_hours",
                           aggfunc="sum", fill_value=0.0).reindex(index=years, columns=flags)

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(years))
    width = 0.8 / len(flags)  # group width split across 10 countries

    cmap = plt.get_cmap("tab20")
    cols = {f: cmap(i % 20) for i, f in enumerate(flags)}

    for i, f in enumerate(flags):
        vals = piv[f].values / 1e6  # millions of hours
        ax.bar(x + i*width - (len(flags)-1)*width/2, vals, width, label=f, color=cols[f], edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=0)
    ax.set_ylabel("Fishing Hours (Millions)", fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', labelsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(savepath, dpi=200)
    plt.show()

plot_clustered_bars(
    sub,
    "",
    OUTDIR / "annual_fishing_hours_top10_excl_china_all_years.png"
)

# -----------------------------
# One chart per year 
# -----------------------------
for y, d in sub.groupby("year"):
    dd = d.groupby("flag", as_index=False)["total_fishing_hours"].sum()
    dd = dd.set_index("flag").reindex(top10_flags).fillna(0.0).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    vals = dd["total_fishing_hours"].values / 1e6
    ax.bar(dd["flag"], vals, color="steelblue", edgecolor="white")
    ax.set_xlabel("Country (flag)")
    ax.set_ylabel("Fishing Hours (Millions)", fontsize = 30, fontweight='bold')
    ax.set_title(f"Annual Fishing Hours by Country ({y})", fontsize=14, fontweight="bold")  
    ax.grid(axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(OUTDIR / f"annual_fishing_hours_by_country_{y}.png", dpi=200)
    plt.close(fig)
print("Saved annual charts in:", OUTDIR)
