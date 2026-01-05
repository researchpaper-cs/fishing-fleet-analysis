import io
import re
import zipfile
from pathlib import Path

import pandas as pd

# ---------- CONFIG ----------
INPUT_DIR = Path("/Users/harinisaravanan/Documents/Python_FishingVessels/Dataset")
OUTPUT_DIR = Path("/Users/harinisaravanan/Documents/Python_FishingVessels/Processed")
MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"
CHUNKSIZE = 250_000  # tune based on RAM; 250k rows per chunk works well
PARQUET_ENGINE = "pyarrow"
# ----------------------------

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("RUNNING data_extract.py")
print("INPUT_DIR:", INPUT_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)

# Identify dataset type from filename
def classify(zip_name: str) -> str | None:
    name = zip_name.lower()
    if name.startswith("mmsi-daily-csvs"):    # e.g., mmsi-daily-csvs-10-v3-2024.zip
        return "dynamic"
    if name.startswith("fleet-monthly-csvs"): # e.g., fleet-monthly-csvs-10-v3-2024.zip
        return "static"
    return None

# Extract year from filename
def get_year(zip_name: str) -> str:
    m = re.search(r"(20\d{2})", zip_name)
    return m.group(1) if m else "unknown"

# Dtypes / schema hints
DYNAMIC_DTYPES = {
    "cell_ll_lat": "float32",
    "cell_ll_lon": "float32",
    "mmsi": "string",
    "hours": "float32",
    "fishing_hours": "float32",
}
STATIC_DTYPES = {
    "year": "Int16",
    "month": "Int8",
    "cell_ll_lat": "float32",
    "cell_ll_lon": "float32",
    "flag": "string",
    "geartype": "string",
    "hours": "float32",
    "fishing_hours": "float32",
    "mmsi_present": "Int32",
}

manifest_rows: list[dict] = []

def iter_csv_members(zf: zipfile.ZipFile):
    """
    Yield (member_name, file_like) for each CSV (or CSV.GZ) inside the zip.
    """
    found = False
    for info in zf.infolist():
        name = info.filename.lower()
        if name.endswith(".csv") or name.endswith(".csv.gz"):
            found = True
            with zf.open(info, "r") as f:
                yield info.filename, io.BytesIO(f.read())
    if not found:
        print("    ⚠️  No CSV/CSV.GZ files found inside this ZIP.")

def safe_parse_dates(file_obj: io.BytesIO) -> list[str]:
    """
    Peek header, return list of date columns to parse (only if present).
    """
    file_obj.seek(0)
    head = pd.read_csv(file_obj, nrows=0, low_memory=False)
    cols = [c.strip().lower() for c in head.columns]
    file_obj.seek(0)
    return ["date"] if "date" in cols else []

def synthesize_date_if_needed(df: pd.DataFrame) -> None:
    """
    If the frame has year+month but not date, create date as the 1st of that month.
    Works with pandas nullable integers.
    """
    if "date" not in df.columns and {"year", "month"}.issubset(df.columns):
        # Robust conversion even if nullable
        try:
            df["date"] = pd.to_datetime(
                dict(year=df["year"], month=df["month"], day=1),
                errors="coerce",
            )
        except Exception:
            df["date"] = pd.to_datetime(
                df["year"].astype("Int64").astype("string") + "-" +
                df["month"].astype("Int64").astype("string") + "-01",
                errors="coerce",
            )

def ensure_expected_columns(df: pd.DataFrame, expected: dict[str, str]) -> None:
    """
    Ensure expected columns exist with the given dtypes; missing ones filled with NA.
    """
    for col, dt in expected.items():
        if col not in df.columns:
            df[col] = pd.Series(pd.NA, index=df.index, dtype=dt)

def process_zip(zip_path: Path):
    dataset_type = classify(zip_path.name)
    if dataset_type is None:
        print(f"⏭️  Skipping (unknown type): {zip_path.name}")
        return

    year = get_year(zip_path.name)
    print(f"\n=== Processing {zip_path.name} → type={dataset_type}, year={year}")

    out_dir = OUTPUT_DIR / dataset_type / f"year={year}"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    files_seen = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = list(iter_csv_members(zf))
        if not members:
            print("  ⚠️  ZIP had no CSV members.")
            return

        for member_name, file_obj in members:
            files_seen += 1
            print(f"  - {member_name}")

            use_dtypes = DYNAMIC_DTYPES if dataset_type == "dynamic" else STATIC_DTYPES
            parse_dates = safe_parse_dates(file_obj)

            # Stream read in chunks
            reader = pd.read_csv(
                file_obj,
                dtype=use_dtypes,
                parse_dates=parse_dates,
                chunksize=CHUNKSIZE,
                low_memory=False,
            )

            chunk_no = 0
            for chunk in reader:
                chunk_no += 1

                # Normalize column names (lowercase / stripped)
                chunk.columns = [c.strip().lower() for c in chunk.columns]

                # If needed, synthesize date from year+month
                synthesize_date_if_needed(chunk)

                # Ensure expected columns exist (missing ones become NA of the right dtype)
                ensure_expected_columns(chunk, use_dtypes)

                # Write out as Parquet (one file per chunk)
                out_file = out_dir / f"{Path(member_name).stem}__part{chunk_no:04d}.parquet"
                try:
                    chunk.to_parquet(out_file, index=False, engine=PARQUET_ENGINE)
                except Exception as e:
                    print(f"    ❌ Parquet write failed for {out_file.name}: {e}")
                    print("    👉 Try: python3 -m pip install pyarrow")
                    raise

                total_rows += len(chunk)
                if chunk_no % 10 == 0:
                    print(f"    ...wrote {chunk_no} chunks so far (~{total_rows:,} rows)")

    if total_rows == 0:
        print("  ⚠️  No rows written from this ZIP.")
        return

    manifest_rows.append({
        "zip_file": zip_path.name,
        "dataset_type": dataset_type,
        "year": year,
        "rows_written": total_rows,
        "parquet_dir": str(out_dir),
    })
    print(f"✅ Done {zip_path.name}: wrote {total_rows:,} rows → {out_dir}")

def main():
    zips = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() == ".zip"])
    if not zips:
        print(f"⚠️  No .zip files found in {INPUT_DIR.resolve()}")
        return

    for zp in zips:
        process_zip(zp)

    if manifest_rows:
        pd.DataFrame(manifest_rows).to_csv(MANIFEST_PATH, index=False)
        print(f"\n🧾 Manifest written to: {MANIFEST_PATH.resolve()}")
    else:
        print("\n⚠️  No files processed. Manifest not created (nothing to record).")

if __name__ == "__main__":
    main()
