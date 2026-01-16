import pandas as pd
import re
from hangul_romanize import Transliter
from hangul_romanize.rule import academic

df = pd.read_csv("CARD_SUBWAY_ALL_merged_with_coordinates.csv")

# Translate the headers to English
df = df.rename(columns={
    "사용일자": "date",
    "노선명": "line",
    "역명": "station_kr",
    "승차총승객수": "boardings",
    "하차총승객수": "alightings",
    "등록일자": "registered_date",
    "lat": "latitude",
    "lng": "longitude"
})

# Drop registered date column
df = df.drop(columns=["registered_date"])

# Convert date to datetime format
df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

# Convert numeric columns to appropriate types
df["boardings"] = df["boardings"].astype("Int64")
df["alightings"] = df["alightings"].astype("Int64")
df["station_code"] = df["station_code"].astype("Int64")

# Romanizer
transliter = Transliter(academic)

def romanize(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text
    return transliter.translit(text).replace("-", " ").title()

# Station romanization
df["station_en"] = df["station_kr"].apply(lambda x: romanize(x) if isinstance(x, str) else x)

# Line romanization (strict)
def romanize_line(s: str) -> str:
    if not isinstance(s, str):
        return s
    # Handle patterns like "2호선"
    m = re.fullmatch(r"(\d+)\s*호선", s.strip())
    if m:
        return f"{m.group(1)} Hoseon"
    return romanize(s)

df["line_rom"] = df["line"].apply(romanize_line)

# Readable English line names (fallback to line_rom if unknown)
line_en_map = {
    "1호선":"Line 1","2호선":"Line 2","3호선":"Line 3","4호선":"Line 4","5호선":"Line 5",
    "6호선":"Line 6","7호선":"Line 7","8호선":"Line 8","9호선":"Line 9",
    "9호선2~3단계":"Line 9 Phase 2~3",
    "공항철도 1호선":"AREX Line 1",
    "경의선":"Gyeongui Line","경춘선":"Gyeongchun Line","경부선":"Gyeongbu Line",
    "경원선":"Gyeongwon Line","경인선":"Gyeongin Line","중앙선":"Jungang Line",
    "분당선":"Bundang Line","수인선":"Suin Line","신림선":"Sillim Line",
    "우이신설선":"Ui–Sinseol Line","서해선":"Seohae Line","과천선":"Gwacheon Line",
    "안산선":"Ansan Line","일산선":"Ilsan Line","장항선":"Janghang Line","경강선":"Gyeonggang Line"
}

def line_to_en(s: str) -> str:
    if not isinstance(s, str):
        return s
    if s in line_en_map:
        return line_en_map[s]
    m = re.fullmatch(r"(\d+)\s*호선", s.strip())
    if m:
        return f"Line {m.group(1)}"
    return romanize_line(s)  # fallback

df["line_en"] = df["line"].apply(line_to_en)

# Features
df["total_flow"] = df["boardings"] + df["alightings"]
df["day_of_week"] = df["date"].dt.day_name()
df["is_weekend"] = df["date"].dt.weekday >= 5
df["month"] = df["date"].dt.to_period("M")

df.to_csv("card_subway_transform_cleaned.csv", index=False, encoding="utf-8-sig")
print(f"card_subway_transform_cleaned.csv ({len(df)} rows) with columns: {['line','line_rom','line_en','station_kr','station_en']}")