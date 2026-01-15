import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------
# Config
# -----------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "dataset" / "card_usage" / "card_subway_transform_cleaned.csv"
OUT_DIR = BASE_DIR / "ml_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATE_COL = "date"
TARGET_COL = "total_flow"

SEQ_LEN = 14
BATCH_SIZE = 128
EPOCHS = 10
LR = 3e-4

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

torch.manual_seed(42)
np.random.seed(42)

# -----------------------
# Load + basic cleaning
# -----------------------
df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COL], low_memory=False)

# numeric cleanup
for col in ["boardings", "alightings", "latitude", "longitude", "station_code", "seoulmetro_code", TARGET_COL]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# is_weekend -> int
if "is_weekend" in df.columns:
    df["is_weekend"] = pd.to_numeric(df["is_weekend"], errors="coerce").fillna(0).astype(int)
else:
    df["is_weekend"] = (df[DATE_COL].dt.weekday >= 5).astype(int)

# Build station_key
code = None
if "seoulmetro_code" in df.columns:
    code = df["seoulmetro_code"]
elif "station_code" in df.columns:
    code = df["station_code"]

if code is not None:
    code_int = code.fillna(-1).astype(int)
    df["station_key"] = np.where(
        code.notna() & (code.astype(float) > 0),
        code_int.astype(str),
        df["line"].astype(str) + "|" + df["station_kr"].astype(str),
    )
else:
    df["station_key"] = df["line"].astype(str) + "|" + df["station_kr"].astype(str)

df = df.sort_values(["station_key", DATE_COL]).reset_index(drop=True)

# Time features
df["day_of_week_num"] = df[DATE_COL].dt.dayofweek
df["day_of_month"] = df[DATE_COL].dt.day
df["week_of_year"] = df[DATE_COL].dt.isocalendar().week.astype(int)

# Keep only needed columns
keep_cols = [
    "station_key", DATE_COL, TARGET_COL,
    "is_weekend", "day_of_week_num", "day_of_month", "week_of_year",
]
HAS_GEO = ("latitude" in df.columns) and ("longitude" in df.columns)
if HAS_GEO:
    keep_cols += ["latitude", "longitude"]

df = df[keep_cols].dropna(subset=[TARGET_COL]).reset_index(drop=True)

# -----------------------
# Time split (and IMPORTANT station filtering)
# -----------------------
split_date = df[DATE_COL].quantile(0.8)
train_df = df[df[DATE_COL] <= split_date].copy()
val_df   = df[df[DATE_COL] >  split_date].copy()

train_stations = set(train_df["station_key"].unique())
val_df = val_df[val_df["station_key"].isin(train_stations)].copy()

print("Train date range:", train_df[DATE_COL].min(), "->", train_df[DATE_COL].max())
print("Val date range  :", val_df[DATE_COL].min(), "->", val_df[DATE_COL].max())
print("Train stations:", len(train_stations), "| Val rows:", len(val_df))

# -----------------------
# Per-station normalization of target
# -----------------------
stats = train_df.groupby("station_key")[TARGET_COL].agg(["mean", "std"]).reset_index()
stats["std"] = stats["std"].replace(0, np.nan)
stats = stats.fillna({"std": 1.0})

train_df = train_df.merge(stats, on="station_key", how="left")
val_df   = val_df.merge(stats, on="station_key", how="left")

# After filtering val stations, mean/std should exist. If not, stop.
if train_df[["mean", "std"]].isna().any().any() or val_df[["mean", "std"]].isna().any().any():
    raise ValueError("Found missing station mean/std after merge. Check station_key consistency.")

train_df["flow_norm"] = (train_df[TARGET_COL] - train_df["mean"]) / train_df["std"]
val_df["flow_norm"]   = (val_df[TARGET_COL]   - val_df["mean"])   / val_df["std"]

# -----------------------
# Feature columns + make them finite + scale non-flow features
# -----------------------
FEATURE_COLS = ["flow_norm", "is_weekend", "day_of_week_num", "day_of_month", "week_of_year"]
if HAS_GEO:
    FEATURE_COLS += ["latitude", "longitude"]

# Coerce numeric + replace inf
for c in FEATURE_COLS:
    train_df[c] = pd.to_numeric(train_df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    val_df[c]   = pd.to_numeric(val_df[c],   errors="coerce").replace([np.inf, -np.inf], np.nan)

# Fill missing values (geo often missing)
train_df[FEATURE_COLS] = train_df[FEATURE_COLS].fillna(0.0)
val_df[FEATURE_COLS]   = val_df[FEATURE_COLS].fillna(0.0)

# Scale non-flow features using train stats (keeps feature magnitudes sane)
SCALE_COLS = [c for c in FEATURE_COLS if c != "flow_norm"]
scale_mu = train_df[SCALE_COLS].mean()
scale_sd = train_df[SCALE_COLS].std().replace(0, 1.0)

train_df[SCALE_COLS] = (train_df[SCALE_COLS] - scale_mu) / scale_sd
val_df[SCALE_COLS]   = (val_df[SCALE_COLS]   - scale_mu) / scale_sd

# Final sanity check
def assert_finite_frame(frame: pd.DataFrame, cols: list[str], name: str):
    bad = ~np.isfinite(frame[cols].to_numpy(dtype=np.float32))
    if bad.any():
        idx = np.argwhere(bad)[0]
        raise ValueError(f"{name} has non-finite values at row {idx[0]} col {cols[idx[1]]}")
assert_finite_frame(train_df, FEATURE_COLS + ["mean", "std"], "train_df")
assert_finite_frame(val_df, FEATURE_COLS + ["mean", "std"], "val_df")

# -----------------------
# Dataset
# -----------------------
class SubwaySeqDataset(Dataset):
    """
    X: (SEQ_LEN, F) for days t-SEQ_LEN+1..t
    y: flow_norm at day t+1
    meta: station_key, date_y, mean, std
    """
    def __init__(self, frame: pd.DataFrame, seq_len: int, feature_cols: list[str]):
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        self.samples = []

        for st, g in frame.groupby("station_key"):
            g = g.sort_values(DATE_COL).reset_index(drop=True)
            if len(g) < seq_len + 1:
                continue

            feats = g[feature_cols].to_numpy(dtype=np.float32)
            y = g["flow_norm"].to_numpy(dtype=np.float32)
            mu = g["mean"].to_numpy(dtype=np.float32)
            sd = g["std"].to_numpy(dtype=np.float32)
            dates = g[DATE_COL].dt.strftime("%Y-%m-%d").astype(str).to_numpy()

            # extra safety
            if not np.isfinite(feats).all() or not np.isfinite(y).all():
                continue

            for t in range(seq_len - 1, len(g) - 1):
                X = feats[t - (seq_len - 1): t + 1]
                target = y[t + 1]
                meta = {
                    "station_key": str(st),
                    "date_y": dates[t + 1],
                    "mean": float(mu[t + 1]),
                    "std": float(sd[t + 1]),
                }
                self.samples.append((X, target, meta))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y, meta = self.samples[idx]
        return torch.from_numpy(X), torch.tensor(y, dtype=torch.float32), meta

def custom_collate(batch):
    Xs, ys, metas = zip(*batch)
    return torch.stack(Xs), torch.stack(ys), list(metas)

train_ds = SubwaySeqDataset(train_df, SEQ_LEN, FEATURE_COLS)
val_ds   = SubwaySeqDataset(val_df,   SEQ_LEN, FEATURE_COLS)

print("Train samples:", len(train_ds))
print("Val samples  :", len(val_ds))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=custom_collate)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)

# Quick batch finiteness check (catches NaN loss root cause)
Xb, yb, _ = next(iter(train_loader))
print("Batch finite:", torch.isfinite(Xb).all().item(), torch.isfinite(yb).all().item())

# -----------------------
# Model
# -----------------------
class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)

model = LSTMRegressor(input_size=len(FEATURE_COLS)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

def eval_loader(loader):
    model.eval()
    preds_norm, trues_norm, metas = [], [], []
    with torch.no_grad():
        for X, y, meta in loader:
            X = X.to(DEVICE, dtype=torch.float32)
            y = y.to(DEVICE, dtype=torch.float32)
            yhat = model(X)

            preds_norm.append(yhat.detach().cpu().numpy())
            trues_norm.append(y.detach().cpu().numpy())
            metas.extend(meta)

    preds_norm = np.concatenate(preds_norm)
    trues_norm = np.concatenate(trues_norm)

    mu = np.array([m["mean"] for m in metas], dtype=np.float32)
    sd = np.array([m["std"] for m in metas], dtype=np.float32)

    preds = preds_norm * sd + mu
    trues = trues_norm * sd + mu

    mae = float(np.mean(np.abs(preds - trues)))
    rmse = float(math.sqrt(np.mean((preds - trues) ** 2)))
    return mae, rmse, preds, trues, metas

# Baseline: predict tomorrow = today (in normalized space)
def eval_persistence(loader):
    preds_norm, trues_norm, metas = [], [], []
    for X, y, meta in loader:
        pred = X[:, -1, 0]  # last timestep's flow_norm
        preds_norm.append(pred.numpy())
        trues_norm.append(y.numpy())
        metas.extend(meta)

    preds_norm = np.concatenate(preds_norm)
    trues_norm = np.concatenate(trues_norm)

    mu = np.array([m["mean"] for m in metas], dtype=np.float32)
    sd = np.array([m["std"] for m in metas], dtype=np.float32)
    preds = preds_norm * sd + mu
    trues = trues_norm * sd + mu

    mae = float(np.mean(np.abs(preds - trues)))
    rmse = float(math.sqrt(np.mean((preds - trues) ** 2)))
    return mae, rmse

print("Persistence baseline (val):", eval_persistence(val_loader))

# -----------------------
# Train
# -----------------------
history = {"train_loss": [], "val_mae": [], "val_rmse": []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0

    for X, y, _ in train_loader:
        X = X.to(DEVICE, dtype=torch.float32)
        y = y.to(DEVICE, dtype=torch.float32)

        optimizer.zero_grad()
        yhat = model(X)
        loss = criterion(yhat, y)

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss. Inputs likely contain NaN/Inf.")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running += float(loss.item())

    train_loss = running / max(1, len(train_loader))
    val_mae, val_rmse, _, _, _ = eval_loader(val_loader)

    history["train_loss"].append(train_loss)
    history["val_mae"].append(val_mae)
    history["val_rmse"].append(val_rmse)

    print(f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.5f} | val_MAE={val_mae:,.2f} | val_RMSE={val_rmse:,.2f}")

# -----------------------
# Save plots + CSV
# -----------------------
val_mae, val_rmse, preds, trues, metas = eval_loader(val_loader)
print("\nFinal Validation")
print(f"MAE  : {val_mae:,.2f}")
print(f"RMSE : {val_rmse:,.2f}")

plt.figure()
plt.plot(history["train_loss"], label="train MSE loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("LSTM Training Loss")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "lstm_training_loss.png", dpi=200)
plt.close()

plt.figure()
idx = np.random.choice(len(preds), size=min(30000, len(preds)), replace=False)
plt.scatter(trues[idx], preds[idx], s=3)
minv = min(trues[idx].min(), preds[idx].min())
maxv = max(trues[idx].max(), preds[idx].max())
plt.plot([minv, maxv], [minv, maxv])
plt.xlabel("Actual next-day total_flow")
plt.ylabel("Predicted next-day total_flow")
plt.title("Predicted vs Actual (Validation)")
plt.tight_layout()
plt.savefig(OUT_DIR / "lstm_pred_vs_actual_scatter.png", dpi=200)
plt.close()

meta_df = pd.DataFrame({
    "station_key": [m["station_key"] for m in metas],
    "date_y": pd.to_datetime([m["date_y"] for m in metas]),
    "pred": preds,
    "true": trues,
})
pick_station = meta_df["station_key"].value_counts().index[0]
st_df = meta_df[meta_df["station_key"] == pick_station].sort_values("date_y")

plt.figure(figsize=(10, 4))
plt.plot(st_df["date_y"], st_df["true"], label="Actual")
plt.plot(st_df["date_y"], st_df["pred"], label="Predicted")
plt.xlabel("Date")
plt.ylabel("Total Flow")
plt.title(f"Next-day Forecast for Station {pick_station}")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "lstm_station_timeseries.png", dpi=200)
plt.close()

meta_df.to_csv(OUT_DIR / "lstm_val_predictions.csv", index=False, encoding="utf-8-sig")

print("\nSaved outputs to:", OUT_DIR)
print("- lstm_training_loss.png")
print("- lstm_pred_vs_actual_scatter.png")
print("- lstm_station_timeseries.png")
print("- lstm_val_predictions.csv")