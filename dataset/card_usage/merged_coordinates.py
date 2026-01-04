import pandas as pd

# Read both CSV files
print("Reading CSV files...")
df_card = pd.read_csv('CARD_SUBWAY_ALL.csv', encoding='utf-8')
df_coords = pd.read_csv('station_code.csv', encoding='utf-8')

print(f"Card data rows: {len(df_card)}")
print(f"Station coordinates rows: {len(df_coords)}")

# Clean column names (remove spaces)
df_coords.columns = df_coords.columns.str.strip()
df_card.columns = df_card.columns.str.strip()

print(f"\nCard data columns: {df_card.columns.tolist()}")
print(f"Coordinates columns: {df_coords.columns.tolist()}")

# Clean station names (remove whitespace)
df_card['역명'] = df_card['역명'].str.strip()
df_coords['station_name(kor)'] = df_coords['station_name(kor)'].str.strip()


# Merge the dataframes on Korean station name
print("\nMerging dataframes...")
merged_df = pd.merge(
    df_card,
    df_coords[['station_name(kor)', 'lat', 'lng', 'station_code', 'seoulmetro_code']],
    left_on='역명',
    right_on='station_name(kor)',
    how='left'
)

print(f"Merged rows: {len(merged_df)}")
print(f"Rows with coordinates: {merged_df['lat'].notna().sum()}")
print(f"Rows without coordinates: {merged_df['lat'].isna().sum()}")
print(f"Match rate: {merged_df['lat'].notna().sum() / len(merged_df) * 100:.2f}%")

# Check which stations didn't match (filter out NaN values)
unmatched = merged_df[merged_df['lat'].isna()]['역명'].dropna().unique()
if len(unmatched) > 0:
    print(f"\nUnmatched stations ({len(unmatched)}):")
    for station in sorted(unmatched):
        print(f"  - {station}")

# Check if there are null station names in the card data
null_stations = merged_df[merged_df['역명'].isna()]
if len(null_stations) > 0:
    print(f"\n Warning: Found {len(null_stations)} rows with null station names")
merged_df = merged_df.drop(columns=['station_name(kor)'])

# Save the merged file
output_file = 'CARD_SUBWAY_ALL_merged_with_coordinates.csv'
merged_df.to_csv(output_file, index=False, encoding='utf-8')

# Show sample of merged data
print("\nSample merged data:")
print(merged_df[['사용일자', '노선명', '역명', 'lat', 'lng', 'station_code']].head(15))

# Show statistics by line
print("\n📊 Match statistics by line:")
line_stats = merged_df.groupby('노선명').agg({
    '역명': 'count',
    'lat': lambda x: x.notna().sum()
}).round(2)
line_stats.columns = ['Total_Rows', 'Matched_Rows']
line_stats['Match_Rate_%'] = (line_stats['Matched_Rows'] / line_stats['Total_Rows'] * 100).round(2)
print(line_stats.sort_values('Match_Rate_%', ascending=False))