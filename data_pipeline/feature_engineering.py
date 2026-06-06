import pandas as pd
import numpy as np
from pymongo import MongoClient
import os, sys
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.logger import log_run

def load_raw_data():
    print("Loading raw data from MongoDB ...")
    client  = MongoClient(os.getenv("MONGO_URI"))
    db      = client[os.getenv("MONGO_DB")]
    records = list(db["raw_features"].find({}, {"_id": 0}))
    client.close()

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"Loaded {len(df)} raw rows")
    return df


def create_features(df):
    # Time-based features
    df["hour"]       = df["datetime"].dt.hour
    df["day"]        = df["datetime"].dt.day
    df["month"]      = df["datetime"].dt.month
    df["weekday"]    = df["datetime"].dt.weekday       # 0=Monday, 6=Sunday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    # Lag features
    df["aqi_lag_1"]  = df["aqi"].shift(1)
    df["aqi_lag_3"]  = df["aqi"].shift(3)
    df["aqi_lag_6"]  = df["aqi"].shift(6)
    df["aqi_lag_24"] = df["aqi"].shift(24)
    df["aqi_lag_48"] = df["aqi"].shift(48)

    # Rolling averages
    df["aqi_rolling_3h"]  = df["aqi"].rolling(3,  min_periods=1).mean().round(2)
    df["aqi_rolling_24h"] = df["aqi"].rolling(24, min_periods=1).mean().round(2)

    # Rate of change
    df["aqi_change_rate"] = df["aqi"].diff(1).round(2)

    # Target columns
    df["target_24h"] = df["aqi"].shift(-24)   # AQI 24 hours from now
    df["target_48h"] = df["aqi"].shift(-48)   # AQI 48 hours from now
    df["target_72h"] = df["aqi"].shift(-72)   # AQI 72 hours from now

    # Drop rows with NaN
    df = df.dropna().reset_index(drop=True)
    print(f"Created features: {len(df)} rows × {len(df.columns)} columns")
    return df


def save_features(df):
    print("Saving engineered features to MongoDB ...")
    client = MongoClient(os.getenv("MONGO_URI"))
    db     = client[os.getenv("MONGO_DB")]
    col    = db["engineered_features"]

    col.drop()  # Replace with fresh data each time
    col.create_index("datetime", unique=True)

    records = df.copy()
    records["datetime"] = records["datetime"].astype(str)
    col.insert_many(records.to_dict("records"))

    print(f"Saved {len(records)} rows to engineered_features")
    client.close()
    return len(records)

if __name__ == "__main__":
    try:
        df      = load_raw_data()
        raw_len = len(df)
        df      = create_features(df)
        saved   = save_features(df)

        log_run(
            script="feature_engineering",
            status="success",
            rows_inserted=saved,
            extra_info={"raw_rows": raw_len, "feature_cols": len(df.columns)}
        )
        print("Feature engineering complete!")

    except Exception as err:
        log_run("feature_engineering", "failed", error_message=str(err))
        print(f"Failed: {err}")
        raise