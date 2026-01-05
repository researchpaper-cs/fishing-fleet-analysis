# ==== Stacked Bar: Countries (Top-10 incl. & excl. China) × Gear Type (Top-6 + OTHERS) ====
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------------------
# Paths (adjust if yours differ)
# ------------------------------------------------------------------------
BASE = Path("/Users/harinisaravanan/Documents/Python_FishingVessels/Processed")
ANALYSIS = BASE / "analysis"
STATIC_GLOB = str(BASE / "static" / "year=*" / "*.parquet")
summary_path = ANALYSIS / "static_summary_country_year.parquet"

# ------------------------------------------------------------------------
# 1) Load summary and compute Top-10 countries (with & without China)
# ------------------------------------------------------------------------
summary_df = pd.read_parquet(summary_path)

# (a) Top-10 excluding CHN and UNKNOWN-CHN
exclude = {"CHN", "UNKNOWN-CHN"}
top10_excl = (
    summary_df[~summary_df["flag"].isin(exclude)]
      .groupby("flag", as_index=False)["total_fishing_hours"]
      .sum()
      .sort_values("total_fishing_hours", ascending=False)
      .head(10)["flag"].tolist()
)
print("Top-10 (excluding China):", top10_excl)

# ------------------------------------------------------------------------
# 2) Aggregate fishing_hours by Country × Gear Type
# ------------------------------------------------------------------------
with duckdb.connect() as con:
    gear_df = con.execute(f"""
        SELECT
            flag,
            geartype AS gear_type,
            SUM(fishing_hours) AS fh
        FROM read_parquet('{STATIC_GLOB}')
        GROUP BY flag, geartype
    """).df()

# ------------------------------------------------------------------------
# 3) Clean and map gear types → Top-6 + OTHERS
# ------------------------------------------------------------------------
def map_gear(raw):
    """Map all messy gear labels to six canonical categories."""
    if pd.isna(raw):
        return "OTHERS"
    g = str(raw).strip().lower().replace("-", "_").replace(" ", "_")

    if any(k in g for k in ["trawler", "trawlers", "otter_trawler", "bottom_trawler", "pair_trawler"]):
        return "TRAWLERS"
    if "longline" in g:
        if "drift" in g or "drifting" in g:
            return "DRIFTING_LONGLINES"
        return "SET_LONGLINES"
    if "gillnet" in g:
        return "SET_GILLNETS"
    if "fixed" in g or "trap" in g or "pot" in g or "dredge" in g:
        return "FIXED_GEAR"
    if g in ["fishing", "general_fishing", "fish"]:
        return "FISHING"
    return "OTHERS"

gear_df["gear_type"] = gear_df["gear_type"].fillna("Unknown")
gear_df["gear_mapped"] = gear_df["gear_type"].apply(map_gear)

KEEP_GEARS = [
    "TRAWLERS",
    "FISHING",
    "DRIFTING_LONGLINES",
    "SET_GILLNETS",
    "SET_LONGLINES",
    "FIXED_GEAR",
]
FULL_ORDER = KEEP_GEARS + ["OTHERS"]

# collapse
agg = gear_df.groupby(["flag", "gear_mapped"], as_index=False)["fh"].sum()

# ------------------------------------------------------------------------
# 4) Helper function to build stacked bar plot
# ------------------------------------------------------------------------
def plot_stacked(top_flags, title, normalize=False):
    sub = agg[agg["flag"].isin(top_flags)]
    wide = sub.pivot_table(
        index="flag", columns="gear_mapped", values="fh", aggfunc="sum", fill_value=0
    )
    wide = wide.reindex(index=[f for f in top_flags if f in wide.index])
    wide = wide.reindex(columns=FULL_ORDER, fill_value=0.0)

    plot_df = wide.copy()
    ylabel = "Fishing Hours"
    if normalize:
        totals = plot_df.sum(axis=1).replace(0, np.nan)
        plot_df = (plot_df.T / totals).T.fillna(0.0)
        ylabel = "Share of Fishing Hours"

    # consistent colors for each gear type
    gear_colors = {
        "TRAWLERS": "#50aaea",          # blue
        "FISHING": "#56759e",           # light blue
        "DRIFTING_LONGLINES": "#B6CEB4",# orange
        "SET_GILLNETS": "#BCA88D",      # light orange
        "SET_LONGLINES": "#C5B0CD",     # green
        "FIXED_GEAR": "#FADA7A",        # light green
        "OTHERS": "#DA6C6C",            # red
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(plot_df))
    x = np.arange(len(plot_df.index))

    for g in FULL_ORDER:
        vals = plot_df[g].values
        ax.bar(
            x,
            vals / (1e6 if not normalize else 1),
            bottom=bottom / (1e6 if not normalize else 1),
            label=g,
            color=gear_colors[g],
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index, rotation=45, ha="right", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_xlabel("Country (flag)",fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Gear Type", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------
# 5) Plot both versions
# ------------------------------------------------------------------------
NORMALIZE = False  # True for percentage stacked

plot_stacked(
    top10_excl,
    title="",
    normalize=NORMALIZE,
)
