# Seoul Metro Temporal Ridership Study
## Overview

This portion of our project explores and predicts subway station ridership patterns in the Seoul Metro system using temporal data from 2015 onwards. By leveraging both classical machine learning (LightGBM) and deep learning (LSTM) approaches, we aim to understand and forecast daily passenger flows at the station level, supporting operational planning and strategic decision-making.

---

## Project Structure
│
├── dataset/
│ └── card_usage/
│ ├── CARD_SUBWAY_ALL_merged_with_coordinates.csv
│ ├── CARD_SUBWAY_ALL.csv
│ ├── CARD_SUBWAY_MONTH_2015.csv ... CARD_SUBWAY_MONTH_202510.csv
│ ├── card_subway_transform_cleaned.csv
│ ├── card_subway_with_timeseries_features.csv
│ ├── station_code.csv
│ ├── merge_csv.py, merged_coordinates.py, transform_dataset.py
│ └── 서울교통공사_노선별 지하철역 정보.json
│   
├── dl/
│ └── train_lstm_timeseries.ipynb
│
├── ml/
│ └── train_lgbm_timeseries.ipynb
│
├── eda/
│ ├── eda.py, plot_map.py
│ └── eda_outputs/
│ ├── line_total_flow.csv, seoul_density_heatmap.html, ...
│
└── README.md

---

## Data

- **Raw Data:** Monthly and yearly ridership CSVs from 2015 to 2025, including station codes and coordinates.
- **Processed Data:** Cleaned and merged datasets with engineered time-series features (lags, rolling means, etc.).
- **Metadata:** Station information and geospatial data for mapping and spatial analysis.

---

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Visualize ridership trends, station popularity, and spatial patterns.
- Analyze temporal patterns (daily, weekly, seasonal effects).

### 2. Feature Engineering
- Generate lag features (e.g., previous day, week, two weeks).
- Compute rolling statistics (mean, std) for recent periods.
- Add calendar features (day of week, month, is_weekend).

### 3. Modeling Approaches

#### LightGBM (ml/train_lgbm_timeseries.ipynb)
- Tabular model using engineered features.
- Hyperparameter tuning and validation.
- Baseline: yesterday’s ridership.

#### LSTM (dl/train_lstm_timeseries.ipynb)
- Sequence model using recent history as input.
- Captures complex temporal dependencies.
- Compared against the same baseline for fair evaluation.

---

## Key Findings
- **Feature engineering** enables strong performance with tree-based models.
- **LSTM models** can capture additional sequential dependencies, but may require more data and tuning to outperform classical models.
- **Operational value:** Accurate next-day predictions support staffing, congestion management, and anomaly detection.

---

## How to Run

1. **Prepare Data:**  
   Use scripts in `dataset/card_usage/` to merge and clean raw data.
2. **EDA:**  
   Run notebooks and scripts in `eda/` for visualization and analysis.
3. **Model Training:**  
   - Run `ml/train_lgbm_timeseries.ipynb` for LightGBM experiments.
   - Run `dl/train_lstm_timeseries.ipynb` for LSTM experiments.

## Authors
IEE3593 Team 7  

## Acknowledgements
- Seoul Metro for open ridership data.
- Open-source libraries: pandas, numpy, LightGBM, PyTorch, matplotlib, scikit-learn.
