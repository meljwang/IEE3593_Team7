# EDA for Seoul subway boarding/alighting dataset (station-month/daily rows)
# Assumes your CSV has columns like:
# date,line,station_kr,boardings,alightings,latitude,longitude,station_code,seoulmetro_code,station_en,total_flow,day_of_week,is_weekend,month

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import platform
import matplotlib
import matplotlib.font_manager as fm

import folium
import branca.colormap as cm

CSV_PATH = "../dataset/card_usage/card_subway_transform_cleaned.csv"  

df = pd.read_csv(CSV_PATH)

# Parse types safely
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["month"] = pd.to_datetime(df["month"].astype(str), errors="coerce").dt.to_period("M")

# Numeric columns
for col in ["boardings", "alightings", "latitude", "longitude", "station_code", "seoulmetro_code"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Ensure boolean
if "is_weekend" in df.columns:
    # Handles True/False strings too
    df["is_weekend"] = df["is_weekend"].astype(str).str.lower().map({"true": True, "false": False}).fillna(df["is_weekend"])

# Create core derived cols (if missing)
if "total_flow" not in df.columns:
    df["total_flow"] = df["boardings"].fillna(0) + df["alightings"].fillna(0)

if "day_of_week" not in df.columns:
    df["day_of_week"] = df["date"].dt.day_name()

# Drop obviously broken rows
df = df.dropna(subset=["date", "station_en", "line"])

print("===== DATA OVERVIEW =====")
print("Rows:", len(df))
print("Stations:", df["station_en"].nunique())
print("Lines:", df["line"].nunique())
print("Date range:", df["date"].min(), "to", df["date"].max())
print("\nMissing values (%):")
print((df.isna().mean() * 100).sort_values(ascending=False).round(2))

# Duplicate check (optional key)
key_cols = ["date", "line", "station_en"]
dupes = df.duplicated(subset=key_cols).sum()
print("\nDuplicates on (date,line,station_en):", dupes)

# Daily total flow
daily_flow = df.groupby("date")["total_flow"].sum().sort_index()

plt.figure(figsize=(12, 4))
plt.plot(daily_flow.index, daily_flow.values)
plt.title("Total Subway Ridership Over Time (Daily Sum)")
plt.xlabel("Date")
plt.ylabel("Passengers")
plt.tight_layout()
plt.show()

# Monthly total flow
monthly_flow = df.groupby("month")["total_flow"].sum().sort_index()

plt.figure(figsize=(12, 4))
plt.plot(monthly_flow.index.astype(str), monthly_flow.values)
plt.title("Total Subway Ridership Over Time (Monthly Sum)")
plt.xlabel("Month")
plt.ylabel("Passengers")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Yearly comparison
df["year"] = df["date"].dt.year
yearly_flow = df.groupby("year")["total_flow"].sum().sort_index()

plt.figure(figsize=(8, 4))
plt.bar(yearly_flow.index.astype(str), yearly_flow.values)
plt.title("Total Ridership by Year")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.tight_layout()
plt.show()


# Boarding vs Alighting + Imbalance
station_means = df.groupby("station_en")[["boardings", "alightings"]].mean().dropna()

plt.figure(figsize=(6, 6))
plt.scatter(station_means["boardings"], station_means["alightings"], alpha=0.5)
mn = min(station_means.min())
mx = max(station_means.max())
plt.plot([mn, mx], [mn, mx], linestyle="--")
plt.title("Avg Boardings vs Avg Alightings (Station Level)")
plt.xlabel("Average boardings")
plt.ylabel("Average alightings")
plt.tight_layout()
plt.show()

df["imbalance"] = df["boardings"] - df["alightings"]
station_imb = df.groupby("station_en")["imbalance"].mean().sort_values()

# Most alighting-heavy (likely employment hubs, tourist attractions, financial centers)
alight_heavy = station_imb.head(15)
plt.figure(figsize=(8, 4))
plt.barh(alight_heavy.index, alight_heavy.values)
plt.title("Stations with Strong Alighting Bias (Avg boardings - alightings)")
plt.xlabel("Average imbalance")
plt.tight_layout()
plt.show()

# Most boarding-heavy (likely residential origins)
board_heavy = station_imb.tail(15)
plt.figure(figsize=(8, 4))
plt.barh(board_heavy.index, board_heavy.values)
plt.title("Stations with Strong Boarding Bias (Avg boardings - alightings)")
plt.xlabel("Average imbalance")
plt.tight_layout()
plt.show()

# Weekday vs Weekend patterns
# Average total flow by day of week
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow_flow = df.groupby("day_of_week")["total_flow"].mean().reindex(dow_order)

plt.figure(figsize=(7, 4))
plt.bar(dow_flow.index, dow_flow.values)
plt.title("Average Total Flow by Day of Week")
plt.xticks(rotation=30)
plt.ylabel("Avg passengers")
plt.tight_layout()
plt.show()

# Weekend vs weekday overall
wk = df[df["is_weekend"] == False]["total_flow"].mean()
we = df[df["is_weekend"] == True]["total_flow"].mean()

plt.figure(figsize=(4, 4))
plt.bar(["Weekday", "Weekend"], [wk, we])
plt.title("Average Total Flow: Weekday vs Weekend")
plt.ylabel("Avg passengers")
plt.tight_layout()
plt.show()

# Station weekend/weekday ratio 
weekday_station = df[df["is_weekend"] == False].groupby("station_en")["total_flow"].mean()
weekend_station = df[df["is_weekend"] == True].groupby("station_en")["total_flow"].mean()

ratio = (weekend_station / weekday_station).dropna()
ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()

print("\nTop 10 'leisure' stations (highest weekend/weekday ratio):")
print(ratio.sort_values(ascending=False).head(10))

print("\nTop 10 'commuter' stations (lowest weekend/weekday ratio):")
print(ratio.sort_values(ascending=True).head(10))

# Metro Line analysis
line_total = df.groupby("line_en")["total_flow"].sum().sort_values()

plt.figure(figsize=(7, 5))
plt.barh(line_total.index, line_total.values)
plt.title("Total Ridership by Line")
plt.xlabel("Total passengers")
plt.tight_layout()
plt.show()

# Map visualization of stations passenger density
station_geo = df.groupby("station_en").agg(
    latitude=("latitude", "mean"),
    longitude=("longitude", "mean"),
    avg_flow=("total_flow", "mean"),
).dropna().reset_index()

m = folium.Map(location=[37.5665, 126.9780], zoom_start=11, tiles="CartoDB positron")

vmin, vmax = station_geo["avg_flow"].min(), station_geo["avg_flow"].max()
cmap = cm.linear.YlOrRd_09.scale(vmin, vmax)

for _, r in station_geo.iterrows():
    folium.CircleMarker(
        location=[r["latitude"], r["longitude"]],
        radius=3 + 7 * (r["avg_flow"] - vmin) / (vmax - vmin + 1e-9),
        color=cmap(r["avg_flow"]),
        fill=True, fill_opacity=0.7, weight=0,
        popup=folium.Popup(f"{r['station_en']}<br>Avg flow: {int(r['avg_flow']):,}", max_width=250)
    ).add_to(m)

cmap.caption = "Average total flow"
cmap.add_to(m)

out_html = "eda_outputs/seoul_stations_map.html"
os.makedirs("eda_outputs", exist_ok=True)
m.save(out_html)
print(f"Saved interactive map -> {out_html}")


# Save EDA summary tables 
out_dir = "eda_outputs"
os.makedirs(out_dir, exist_ok=True)

station_total.head(200).to_csv(os.path.join(out_dir, "top_stations_total_flow.csv"))
station_imb.to_csv(os.path.join(out_dir, "station_imbalance_mean.csv"))
line_total.to_csv(os.path.join(out_dir, "line_total_flow.csv"))
ratio.sort_values(ascending=False).to_csv(os.path.join(out_dir, "weekend_weekday_ratio_by_station.csv"))

print(f"\nSaved summary tables to: {out_dir}/")
