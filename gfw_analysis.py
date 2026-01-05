# gfw_analysis_v5_mac.py
# Memory-safe analysis for your Processed Parquet outputs on macOS.
# - STATIC (fleet-monthly-*) is processed year-by-year.
# - DYNAMIC (mmsi-daily-*) can be processed for selected years to stay RAM-safe.

from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

# =================== CONFIG ======================
# Root folder that contains: Processed/static/year=YYYY and Processed/dynamic/year=YYYY
ROOT = Path("/Users/harinisaravanan/Documents/Python_FishingVessels")
OUT  = ROOT / "Processed" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

# Which years to read from STATIC parquet (None => all)
YEARS_STATIC = None         # e.g., [2022, 2023]

# Which years to read from DYNAMIC parquet (empty list => skip)
DYNAMIC_YEARS = []          # e.g., [2024]

# Optional clustering (disabled by default)
USE_CLUSTERING = False
K_CLUSTERS = 5
# =================================================


# ----------------- UTILITIES ---------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """
    Vectorized Haversine distance in kilometers.
    Fixed: use **2 on sine terms (classic formula).
    """
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon / 2.0) ** 2))
    return 2 * R * np.arcsin(np.sqrt(a))


def _iter_year_dirs(base: Path, years=None):
    """
    Yield (year_int, path) for folders like base/year=YYYY.
    If years is None => all; else filter to provided list.
    """
    for p in sorted(base.glob("year=*")):
        if p.is_dir():
            try:
                y = int(p.name.split("=", 1)[1])
            except Exception:
                continue
            if years is None or y in years:
                yield y, p


# --------------- LOAD PARQUET (STATIC) -----------
def load_static_yearwise(columns=None, years=None) -> pd.DataFrame:
    """
    Read static parquet year-by-year to avoid schema merge issues.
    Injects 'year' from folder, and coerces 'month' to Int64 if present.
    """
    base = ROOT / "Processed" / "static"
    frames = []
    for year, ydir in _iter_year_dirs(base, years=years):
        dataset = ds.dataset(str(ydir), format="parquet")
        cols = None if columns is None else [c for c in columns if c != "year"]
        tbl = dataset.to_table(columns=cols)
        df = tbl.to_pandas(types_mapper=pd.ArrowDtype)

        df["year"] = pd.Series(year, index=df.index, dtype="Int64")
        if "month" in df.columns:
            df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
        frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# -------- LOAD PARQUET (DYNAMIC, one-year) -------
def load_dynamic_one_year(year: int, columns=None) -> pd.DataFrame:
    """
    Read ONE dynamic year to keep memory low. Adds 'year' from folder and 'month' from 'date' if present.
    """
    base = ROOT / "Processed" / "dynamic" / f"year={year}"
    if not base.exists():
        return pd.DataFrame()
    dataset = ds.dataset(str(base), format="parquet")
    tbl = dataset.to_table(columns=columns)
    df = tbl.to_pandas(types_mapper=pd.ArrowDtype)

    df["year"] = pd.Series(year, index=df.index, dtype="Int64")

    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = dt.dt.month.astype("Int64")
        del dt

    return df


# ----------- CENTROIDS & RADII (STATIC) ----------
def monthly_centroid_static(df: pd.DataFrame, per_flag=False, weight="fishing_hours") -> pd.DataFrame:
    group_cols = ["year", "month"]
    if per_flag:
        group_cols.append("flag")

    w = pd.to_numeric(df.get(weight, 0.0), errors="coerce").fillna(0.0).astype("float64")
    g = df.assign(
        w=w,
        wlat=pd.to_numeric(df["cell_ll_lat"], errors="coerce").astype("float64") * w,
        wlon=pd.to_numeric(df["cell_ll_lon"], errors="coerce").astype("float64") * w,
    )
    cent = (
        g.groupby(group_cols, as_index=False)
         .agg(lat_num=("wlat", "sum"), lon_num=("wlon", "sum"), wsum=("w", "sum"))
    )
    cent["lat_centroid"] = np.where(cent["wsum"] > 0, cent["lat_num"] / cent["wsum"], np.nan)
    cent["lon_centroid"] = np.where(cent["wsum"] > 0, cent["lon_num"] / cent["wsum"], np.nan)
    return cent[group_cols + ["lat_centroid", "lon_centroid"]]


def monthly_radius_static(df: pd.DataFrame, centroids: pd.DataFrame, per_flag=False, weight="fishing_hours") -> pd.DataFrame:
    group_cols = ["year", "month"]
    if per_flag:
        group_cols.append("flag")

    merged = df.merge(centroids, on=group_cols, how="left")
    lat  = pd.to_numeric(merged["cell_ll_lat"], errors="coerce").astype("float64")
    lon  = pd.to_numeric(merged["cell_ll_lon"], errors="coerce").astype("float64")
    latc = pd.to_numeric(merged["lat_centroid"], errors="coerce").astype("float64")
    lonc = pd.to_numeric(merged["lon_centroid"], errors="coerce").astype("float64")

    merged["dist_km"] = haversine_km(lat, lon, latc, lonc)

    def _agg_block(x: pd.DataFrame) -> pd.Series:
        w = pd.to_numeric(x.get(weight, 0.0), errors="coerce").fillna(0.0).to_numpy()
        d = x["dist_km"].to_numpy()
        wsum = w.sum()
        radius = (d @ w) / wsum if wsum > 0 else np.nan
        if len(d) == 0 or np.all(np.isnan(d)):
            return pd.Series({"radius_km": np.nan, "max_dist_km": np.nan,
                              "furthest_lat": np.nan, "furthest_lon": np.nan})
        imax = int(np.nanargmax(d))
        return pd.Series({
            "radius_km": float(radius),
            "max_dist_km": float(d[imax]),
            "furthest_lat": float(x["cell_ll_lat"].iloc[imax]),
            "furthest_lon": float(x["cell_ll_lon"].iloc[imax]),
        })

    out = merged.groupby(group_cols, as_index=False).apply(_agg_block)
    # Pandas >=2.2 puts group keys in index; normalize:
    if isinstance(out.index, pd.MultiIndex):
        out = out.droplevel(list(range(out.index.nlevels))).reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
    return out


def centroid_shift_km(centroids: pd.DataFrame, per_flag=False) -> pd.DataFrame:
    c = centroids.copy()
    group_cols = ["flag"] if per_flag else []
    c["yyyymm"] = c["year"].astype("Int64").astype(int) * 100 + c["month"].astype("Int64").astype(int)
    c = c.sort_values(group_cols + ["yyyymm"])

    def _lag_shift(x: pd.DataFrame) -> pd.DataFrame:
        x = x.sort_values("yyyymm")
        lat_prev = x["lat_centroid"].shift(1)
        lon_prev = x["lon_centroid"].shift(1)
        x["shift_km"] = haversine_km(x["lat_centroid"], x["lon_centroid"], lat_prev, lon_prev)
        return x

    if group_cols:
        c = c.groupby(group_cols, group_keys=False).apply(_lag_shift)
    else:
        c = _lag_shift(c)
    return c.drop(columns=["yyyymm"])


# -------- VESSELS vs FISHING HOURS (STATIC) ------
def vessels_vs_hours(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["year", "month"], as_index=False)
          .agg(vessels=("mmsi_present", "sum"),
               fishing_hours=("fishing_hours", "sum"))
    )
    agg["vessels"] = pd.to_numeric(agg["vessels"], errors="coerce").astype("Int64")
    return agg


# --------------- COUNTRY SUMMARIES ----------------
def summarize_countries(df_static: pd.DataFrame, flags: list[str], years=None) -> pd.DataFrame:
    sub = df_static[df_static["flag"].isin(flags)].copy()
    if years:
        sub = sub[sub["year"].isin(years)]

    cent  = monthly_centroid_static(sub, per_flag=True)
    rad   = monthly_radius_static(sub, cent, per_flag=True)
    shift = centroid_shift_km(cent, per_flag=True)

    sums = (
        sub.groupby(["year", "month", "flag"], as_index=False)
           .agg(fishing_hours=("fishing_hours", "sum"),
                vessels=("mmsi_present", "sum"))
    )

    out = (
        cent.merge(rad, on=["year", "month", "flag"], how="left")
            .merge(shift[["year", "month", "flag", "shift_km"]],
                   on=["year", "month", "flag"], how="left")
            .merge(sums, on=["year", "month", "flag"], how="left")
    )
    return out


# ----------------- CLUSTERING (opt) ---------------
def cluster_countries_yearly(df_static: pd.DataFrame, k=5, years=None):
    if not USE_CLUSTERING:
        raise RuntimeError("Clustering disabled (set USE_CLUSTERING=True to enable).")
    try:
        from sklearn.cluster import KMeans
    except Exception as e:
        raise RuntimeError("scikit-learn not available. Install it and retry.") from e

    sub = df_static.copy()
    if years:
        sub = sub[sub["year"].isin(years)]

    feat = (
        sub.groupby(["flag", "year"], as_index=False)
           .agg(fishing_hours=("fishing_hours", "sum"),
                vessels=("mmsi_present", "sum"),
                cells=("cell_ll_lat", "count"))
    )
    X = feat[["fishing_hours", "vessels", "cells"]].fillna(0.0).to_numpy()
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    feat["cluster"] = km.fit_predict(X)
    return feat, km


# ==================== MAIN =======================
if __name__ == "__main__":
    # ---- STATIC: Full pipeline ----
    print("Loading STATIC (fleet-monthly) year-by-year…")
    static_cols = ["month","cell_ll_lat","cell_ll_lon","flag","geartype","hours","fishing_hours","mmsi_present"]
    static_df = load_static_yearwise(columns=static_cols, years=YEARS_STATIC)
    print(f"  static rows: {len(static_df):,}")

    if static_df.empty:
        print("⚠️  No static data found under Processed/static/year=YYYY")
    else:
        # Global monthly centroid / radius / shift
        cent_all  = monthly_centroid_static(static_df, per_flag=False)
        rad_all   = monthly_radius_static(static_df, cent_all, per_flag=False)
        shift_all = centroid_shift_km(cent_all, per_flag=False)
        summary_all = (
            cent_all.merge(rad_all, on=["year","month"], how="left")
                    .merge(shift_all[["year","month","shift_km"]], on=["year","month"], how="left")
        )
        summary_all.to_csv(OUT / "global_monthly_centroid_radius_shift.csv", index=False)
        print("✓ global_monthly_centroid_radius_shift.csv")

        # Vessels vs hours
        vh = vessels_vs_hours(static_df)
        vh.to_csv(OUT / "vessels_vs_fishing_hours.csv", index=False)
        print("✓ vessels_vs_fishing_hours.csv")

        # Country comparison (edit flags if you want)
        FLAGS = ["ARG", "CHN", "ESP"]
        country_monthly = summarize_countries(static_df, flags=FLAGS, years=YEARS_STATIC)
        country_monthly.to_csv(OUT / "country_monthly_summary.csv", index=False)
        print(f"✓ country_monthly_summary.csv (flags: {FLAGS})")

        # Optional clustering
        if USE_CLUSTERING:
            try:
                feat_yearly, model = cluster_countries_yearly(static_df, k=K_CLUSTERS, years=YEARS_STATIC)
                feat_yearly.to_csv(OUT / "country_yearly_clusters.csv", index=False)
                print(f"✓ country_yearly_clusters.csv (k={K_CLUSTERS})")
            except Exception as e:
                print("Clustering skipped:", e)

    # ---- DYNAMIC: Process per selected year(s) only (optional) ----
    if DYNAMIC_YEARS:
        print(f"Processing DYNAMIC years: {DYNAMIC_YEARS}")
        dyn_cols = ["date","cell_ll_lat","cell_ll_lon","mmsi","hours","fishing_hours"]
        for y in DYNAMIC_YEARS:
            print(f"  Loading dynamic year {y} ...")
            dy = load_dynamic_one_year(y, columns=dyn_cols)
            if dy.empty or "mmsi" not in dy.columns:
                print(f"  Year {y}: no dynamic data found; skipping.")
                continue

            # Monthly centroid (weighted by fishing_hours) for that year
            w = pd.to_numeric(dy["fishing_hours"], errors="coerce").fillna(0.0).astype("float64")
            dy2 = dy.assign(
                w=w,
                wlat=pd.to_numeric(dy["cell_ll_lat"], errors="coerce").astype("float64") * w,
                wlon=pd.to_numeric(dy["cell_ll_lon"], errors="coerce").astype("float64") * w,
            )
            dyn_cent = (dy2.groupby(["year","month"], as_index=False)
                          .agg(lat_num=("wlat","sum"), lon_num=("wlon","sum"), wsum=("w","sum")))
            dyn_cent["lat_centroid"] = np.where(dyn_cent["wsum"] > 0, dyn_cent["lat_num"]/dyn_cent["wsum"], np.nan)
            dyn_cent["lon_centroid"] = np.where(dyn_cent["wsum"] > 0, dyn_cent["lon_num"]/dyn_cent["wsum"], np.nan)
            dyn_cent = dyn_cent[["year","month","lat_centroid","lon_centroid"]]

            # Example: top 2 MMSIs by row count in that year
            top_mmsi = dy["mmsi"].value_counts().head(2).index.tolist()
            if top_mmsi:
                ships = dy[dy["mmsi"].isin(top_mmsi)].copy()
                ships = ships.merge(dyn_cent, on=["year","month"], how="left")
                ships["dist_to_centroid_km"] = haversine_km(
                    pd.to_numeric(ships["cell_ll_lat"], errors="coerce").astype("float64"),
                    pd.to_numeric(ships["cell_ll_lon"], errors="coerce").astype("float64"),
                    pd.to_numeric(ships["lat_centroid"], errors="coerce").astype("float64"),
                    pd.to_numeric(ships["lon_centroid"], errors="coerce").astype("float64")
                )
                ship_monthly = (
                    ships.groupby(["mmsi","year","month"], as_index=False)
                         .agg(avg_dist_km=("dist_to_centroid_km","mean"),
                              fishing_hours=("fishing_hours","sum"))
                )
                ship_monthly.to_csv(OUT / f"example_two_ships_monthly_distance_{y}.csv", index=False)
                print(f"  ✓ example_two_ships_monthly_distance_{y}.csv (MMSIs: {top_mmsi})")
            else:
                print(f"  Year {y}: no MMSI values; ship example skipped.")

            # cleanup
            del dy, dy2, dyn_cent
            if 'ships' in locals(): del ships
            if 'ship_monthly' in locals(): del ship_monthly
    else:
        print("Dynamic processing skipped (set DYNAMIC_YEARS = [YYYY] to enable).")

    print(f"\nAll outputs written to: {OUT.resolve()}")
