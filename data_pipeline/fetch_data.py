import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pymongo import MongoClient
import os, sys
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.logger import log_run

LATITUDE  = 33.6844
LONGITUDE = 73.0479
TODAY     = datetime.now().strftime("%Y-%m-%d")


def fetch_aqi_data(start_date=None, end_date=None):
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = TODAY

    if end_date > TODAY:
        end_date = TODAY

    print(f"Fetching {start_date} → {end_date} ...")

    aq_resp = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude":   LATITUDE,
            "longitude":  LONGITUDE,
            "hourly":     "pm2_5,pm10,us_aqi",
            "start_date": start_date,
            "end_date":   end_date,
        },
        timeout=30
    )
    aq_resp.raise_for_status()
    aq = aq_resp.json()

    five_days_ago = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
    if end_date <= five_days_ago:
        wx_url = "https://archive-api.open-meteo.com/v1/archive"
    else:
        wx_url = "https://api.open-meteo.com/v1/forecast"

    wx_resp = requests.get(
        wx_url,
        params={
            "latitude":   LATITUDE,
            "longitude":  LONGITUDE,
            "hourly":     "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
            "start_date": start_date,
            "end_date":   end_date,
        },
        timeout=30
    )
    wx_resp.raise_for_status()
    wx = wx_resp.json()

    df = pd.DataFrame({
        "datetime":      aq["hourly"]["time"],
        "pm2_5":         aq["hourly"]["pm2_5"],
        "pm10":          aq["hourly"]["pm10"],
        "aqi":           aq["hourly"]["us_aqi"],
        "temperature":   wx["hourly"]["temperature_2m"],
        "humidity":      wx["hourly"]["relative_humidity_2m"],
        "wind_speed":    wx["hourly"]["wind_speed_10m"],
        "precipitation": wx["hourly"]["precipitation"],
    })

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.ffill().dropna(subset=["aqi"])
    df["aqi"]   = df["aqi"].astype(float).round(1)
    df["pm2_5"] = df["pm2_5"].astype(float).round(2)
    df["pm10"]  = df["pm10"].astype(float).round(2)
    df = df.reset_index(drop=True)

    print(f"Fetched {len(df)} rows")
    return df, start_date, end_date


def store_in_mongodb(df):
    print("Connecting to MongoDB ...")
    client = MongoClient(os.getenv("MONGO_URI"))
    db     = client[os.getenv("MONGO_DB")]
    col    = db["raw_features"]
    col.create_index("datetime", unique=True)

    records  = df.copy()
    records["datetime"] = records["datetime"].astype(str)
    records  = records.to_dict("records")

    inserted = 0
    skipped  = 0
    try:
        result = col.insert_many(records, ordered=False)
        inserted = len(result.inserted_ids)
    except Exception as e:
        inserted = e.details.get("nInserted", 0)
        skipped  = len(records) - inserted

    print(f"Inserted {inserted} | Skipped {skipped} duplicates")
    client.close()
    return inserted, skipped

if __name__ == "__main__":
    try:
        df, s, e = fetch_aqi_data()
        print(df[["datetime","aqi","pm2_5","temperature"]].head())
        inserted, skipped = store_in_mongodb(df)
        log_run("fetch_data", "success", inserted, skipped, f"{s} to {e}",
                extra_info={"total_fetched": len(df)})
        print("Done!")
    except Exception as err:
        log_run("fetch_data", "failed", error_message=str(err))
        print(f"Failed: {err}")
        raise