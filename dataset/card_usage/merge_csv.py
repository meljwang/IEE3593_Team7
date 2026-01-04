import pandas as pd
from pathlib import Path

files = sorted(Path(".").glob("CARD_SUBWAY_MONTH_*.csv"))
dfs = []

for f in files:
    print("Reading:", f.name)
    df = pd.read_csv(
            f,
            encoding="utf-8",
            sep=",",
            quotechar='"',
            header=0,
            usecols=[0,1,2,3,4,5],   
            dtype={
                '사용일자': str,
                '노선명': str,
                '역명': str,
                '승차총승객수': 'Int64',
                '하차총승객수': 'Int64',
                '등록일자': str
            }
    )

    # Clean
    df['사용일자'] = df['사용일자'].astype(str).str.strip()
    df = df[df['사용일자'].ne('').astype(bool)]
    dfs.append(df)

# Concatenate and sort
merged = pd.concat(dfs, ignore_index=True).sort_values('사용일자').reset_index(drop=True)

# Save
merged.to_csv("CARD_SUBWAY_ALL.csv", index=False, encoding="utf-8-sig")
print(f"Done -> CARD_SUBWAY_ALL.csv ({len(merged)} rows)")
print(f"Columns: {merged.columns.tolist()}")
print("Sample:", merged[['사용일자','노선명','역명']].head(5).to_dict(orient='records'))