import pandas as pd
import folium
from folium.plugins import HeatMap
from pathlib import Path

csv_path = Path("../dataset/card_usage/card_subway_transform_cleaned.csv")

df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)

df['latitude'] = pd.to_numeric(df.get('latitude'), errors='coerce')
df['longitude'] = pd.to_numeric(df.get('longitude'), errors='coerce')
df['date'] = pd.to_datetime(df.get('date'), errors='coerce')
df['total_flow'] = (df.get('boardings', 0).fillna(0) + df.get('alightings', 0).fillna(0))

# Average daily flow per station 
daily_station = (
    df.dropna(subset=['latitude','longitude','date'])
      .groupby(['station_en','latitude','longitude','date'], as_index=False)['total_flow']
      .sum()
)
pts = (
    daily_station.groupby(['station_en','latitude','longitude'], as_index=False)['total_flow']
    .mean()
    .rename(columns={'total_flow':'avg_daily_flow'})
)

m = folium.Map(location=[37.5665, 126.9780], zoom_start=11, tiles="CartoDB positron")


heat_data = pts[['latitude','longitude','avg_daily_flow']].values.tolist()
HeatMap(
    heat_data,
    min_opacity=0.3,
    radius=12,
    blur=18,
    gradient={  
        0.0: '#ffffff',
        0.4: '#ffeda0',
        0.6: '#feb24c',
        0.8: '#f03b20',
        1.0: '#bd0026'
    }
).add_to(m)

for _, r in pts.iterrows():
    folium.CircleMarker(
        location=[r['latitude'], r['longitude']],
        radius=2, color='#333', fill=True, fill_opacity=0.6,
        popup=f"{r['station_en']}<br>Avg daily flow: {int(r['avg_daily_flow']):,}"
    ).add_to(m)

out_html = Path("eda_outputs/seoul_density_heatmap.html")
out_html.parent.mkdir(exist_ok=True)
m.save(str(out_html))
print(f"Saved interactive map -> {out_html}")