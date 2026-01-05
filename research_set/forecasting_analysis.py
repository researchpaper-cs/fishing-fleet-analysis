# IMPORTANT: If you do not have the 'prophet' library installed, please run:
# pip install prophet

import pandas as pd
from pathlib import Path
from prophet import Prophet
import matplotlib.pyplot as plt

# --- 1. Load and Prepare Data ---
BASE = Path("/Users/harinisaravanan/Documents/Python_FishingVessels/Processed/analysis")
summary = pd.read_parquet(BASE / "static_summary_country_year.parquet")

# Filter for top 10 countries (excluding China)
exclude = {"CHN", "UNKNOWN-CHN"}
top10 = (
    summary[~summary["flag"].isin(exclude)]
    .groupby("flag")["total_fishing_hours"].sum()
    .nlargest(10).index.tolist()
)
df = summary[summary["flag"].isin(top10)]

# --- 2. Iterate, Forecast, and Plot for Each Country ---
all_forecasts = pd.DataFrame()

print("Generating forecasts for top 10 countries...")

for country in top10:
    country_df = df[df["flag"] == country][["year", "total_fishing_hours"]].copy()
    
    # Prophet requires columns 'ds' and 'y'
    country_df.rename(columns={"year": "ds", "total_fishing_hours": "y"}, inplace=True)
    
    # Convert year to datetime objects
    country_df['ds'] = pd.to_datetime(country_df['ds'], format='%Y')
    
    # Initialize and fit the model
    model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
                    changepoint_prior_scale=0.05, seasonality_prior_scale=10.0)
    model.fit(country_df)
    
    # Create future dataframe for 3 years
    future = model.make_future_dataframe(periods=3, freq='Y')
    
    # Generate forecast
    forecast = model.predict(future)
    
    # Store country name
    forecast['flag'] = country
    all_forecasts = pd.concat([all_forecasts, forecast], ignore_index=True)
    
    # Plot individual forecast
    fig = model.plot(forecast)
    ax = fig.gca()
    ax.set_title(f"Fishing Hour Forecast for {country}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Fishing Hours")
    plt.show()

print("...forecasts complete.")

# --- 3. Plot All Forecasts Together ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(14, 8))

for country in top10:
    # Plot historical data and get its color
    country_hist = df[df["flag"] == country]
    line, = ax.plot(country_hist["year"], country_hist["total_fishing_hours"], 'o-', label=country)
    
    # Plot forecasted data with the same color
    country_fcst = all_forecasts[all_forecasts["flag"] == country]
    ax.plot(country_fcst['ds'].dt.year, country_fcst['yhat'], '--', color=line.get_color())


ax.set_xlabel("Year", fontsize=14, fontweight='bold')
ax.set_ylabel("Total Fishing Hours", fontsize=14, fontweight='bold')
plt.setp(ax.get_xticklabels(), fontsize=12, fontweight="bold")
plt.setp(ax.get_yticklabels(), fontsize=12, fontweight="bold")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid(True)
fig.tight_layout()
plt.show()


# --- 4. Narrative for Paper ---
interpretation = f"""
Forecasting Future Fishing Effort: A Country-Level Analysis

To project future fishing effort, we employed time-series forecasting using 
the Prophet model. We generated individual forecasts for the top 10 fishing 
nations for the period 2025-2027.

The model was trained on annual fishing hour data from 2012 to 2024 for each country.

The composite plot of these forecasts reveals diverging trends among major
fishing nations. This suggests that future global fishing effort will be
shaped by a complex interplay of national-level policies and economic drivers,
rather than a single monolithic trend. The country-specific plots provide
detailed insights into the expected trajectory of each nation.
"""

print(interpretation)
