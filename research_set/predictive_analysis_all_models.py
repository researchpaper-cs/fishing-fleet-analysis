import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# --- 1. Load and Preprocess Data ---
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

# Prepare data for modeling
df_model = df[["year", "flag", "tonnage", "total_fishing_hours"]].copy()
df_model = pd.get_dummies(df_model, columns=["flag"], drop_first=True)

# Features (X) and Target (y)
X = df_model.drop("total_fishing_hours", axis=1)
y = df_model["total_fishing_hours"]

# --- 2. Split Data (Time-Based) ---
X_train = X[X["year"] < 2022]
y_train = y[X["year"] < 2022]
X_test = X[X["year"] >= 2022]
y_test = y[X["year"] >= 2022]

# --- 3. Train and Compare Models ---
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {"model": model, "y_pred": y_pred, "r2": r2, "rmse": np.sqrt(mse)}
    
    print(f"--- {name} ---")
    print(f"  R-squared: {r2:.4f}")
    print(f"  RMSE: {np.sqrt(mse):,.2f}")

# --- 4. Visualize Results for All Models ---
for name, result in results.items():
    y_pred = result["y_pred"]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(y_test / 1e6, y_pred / 1e6, alpha=0.7, edgecolors="k", label="Predictions")
    ax.plot([y_test.min() / 1e6, y_test.max() / 1e6], [y_test.min() / 1e6, y_test.max() / 1e6], 'r--', lw=2, label="Perfect Prediction")

    ax.set_xlabel("Actual Fishing Hours (Millions)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Predicted Fishing Hours (Millions)", fontsize=12, fontweight='bold')
    ax.set_title(f"Model Performance: {name}")
    ax.legend()
    ax.grid(True)

    # Add interpretation text for the paper
    r2_val = result['r2']
    rmse_val = result['rmse']
    stats_text = f"R-squared: {r2_val:.3f}\nRMSE: {rmse_val / 1e6:.3f}M hours"

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    fig.tight_layout()
    plt.show()
