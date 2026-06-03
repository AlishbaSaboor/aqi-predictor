"""
backfill.py
Run this ONCE to fill MongoDB with 6 months of historical data.
Do not run it again after the first time.
"""
from fetch_data import fetch_aqi_data, store_in_mongodb
from datetime import datetime, timedelta
import sys, os, time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.logger import log_run


def backfill(start_date="2025-01-01"):
    end_date    = datetime.now().strftime("%Y-%m-%d")
    current     = datetime.strptime(start_date, "%Y-%m-%d")
    end         = datetime.strptime(end_date,   "%Y-%m-%d")
    total_new   = 0
    chunk_num   = 0

    print(f"🚀 Backfill: {start_date} → {end_date}")
    print("This takes 5–10 minutes. Do not close the terminal.\n")

    while current < end:
        chunk_end = min(current + timedelta(days=30), end)
        s = current.strftime("%Y-%m-%d")
        e = chunk_end.strftime("%Y-%m-%d")
        chunk_num += 1
        print(f"── Chunk {chunk_num}: {s} → {e}")

        try:
            df, _, _ = fetch_aqi_data(start_date=s, end_date=e)
            inserted, skipped = store_in_mongodb(df)
            total_new += inserted
            log_run("backfill", "success", inserted, skipped, f"{s} to {e}")
        except Exception as err:
            print(f"  ⚠️ Error: {err} — skipping chunk")
            log_run("backfill", "failed", error_message=str(err), date_range=f"{s} to {e}")

        current = chunk_end + timedelta(days=1)
        time.sleep(1)  # Be polite to the API

    print(f"\n🎉 Backfill complete!")
    print(f"   Chunks processed : {chunk_num}")
    print(f"   New rows inserted: {total_new}")
    print("   Check MongoDB Atlas → raw_features collection.")


if __name__ == "__main__":
    backfill(start_date="2025-01-01")