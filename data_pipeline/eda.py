import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

print("🔗 Loading data ...")
client  = MongoClient(os.getenv("MONGO_URI"))
records = list(client[os.getenv("MONGO_DB")]["engineered_features"].find({}, {"_id": 0}))
client.close()

df = pd.DataFrame(records)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)
print(f"✅ Loaded {len(df)} rows\n")

print("BASIC STATISTICS:")
print(df[["aqi","pm2_5","pm10","temperature","humidity","wind_speed"]].describe().round(2))
print("\nMissing values:")
print(df.isnull().sum())

# ── Chart 1: AQI over time ─────────────────────────────────────
plt.figure(figsize=(14,4))
plt.plot(df["datetime"], df["aqi"], lw=0.7, color="#2980b9", alpha=0.8)
for y,c,lbl in [(50,"green","Good(50)"),(100,"orange","Moderate(100)"),(150,"red","Unhealthy(150)")]:
    plt.axhline(y, color=c, ls="--", lw=1, label=lbl)
plt.title("AQI Over Time — Islamabad 2025–2026", fontsize=14)
plt.xlabel("Date"); plt.ylabel("AQI"); plt.legend(); plt.tight_layout()
plt.savefig("eda_1_timeseries.png", dpi=150); plt.show()
print("✅ Saved eda_1_timeseries.png")

# ── Chart 2: Average AQI by hour of day ───────────────────────
plt.figure(figsize=(10,4))
ha = df.groupby("hour")["aqi"].mean()
colors = ["#e74c3c" if v>100 else "#f39c12" if v>50 else "#27ae60" for v in ha]
ha.plot(kind="bar", color=colors, edgecolor="white")
plt.title("Average AQI by Hour of Day", fontsize=14)
plt.xlabel("Hour (0=midnight)"); plt.ylabel("Avg AQI"); plt.xticks(rotation=0); plt.tight_layout()
plt.savefig("eda_2_by_hour.png", dpi=150); plt.show()
print("✅ Saved eda_2_by_hour.png")

# ── Chart 3: Average AQI by month ─────────────────────────────
plt.figure(figsize=(10,4))
mnames = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
ma = df.groupby("month")["aqi"].mean()
ma.index = [mnames[i-1] for i in ma.index]
ma.plot(kind="bar", color="#9b59b6", edgecolor="white")
plt.title("Average AQI by Month", fontsize=14)
plt.xlabel("Month"); plt.ylabel("Avg AQI"); plt.xticks(rotation=45); plt.tight_layout()
plt.savefig("eda_3_by_month.png", dpi=150); plt.show()
print("✅ Saved eda_3_by_month.png")

# ── Chart 4: Correlation heatmap ──────────────────────────────
cols = ["aqi","pm2_5","pm10","temperature","humidity","wind_speed","hour","month","aqi_lag_24"]
corr = df[cols].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            mask=np.triu(np.ones_like(corr,dtype=bool)), square=True, linewidths=0.5)
plt.title("Feature Correlation Heatmap", fontsize=14); plt.tight_layout()
plt.savefig("eda_4_correlation.png", dpi=150); plt.show()
print("✅ Saved eda_4_correlation.png")

# ── Chart 5: AQI distribution ─────────────────────────────────
plt.figure(figsize=(10,4))
plt.hist(df["aqi"], bins=60, color="#1abc9c", edgecolor="white", alpha=0.85)
plt.axvline(df["aqi"].mean(),   color="red",  ls="--", label=f"Mean {df['aqi'].mean():.1f}")
plt.axvline(df["aqi"].median(), color="blue", ls="--", label=f"Median {df['aqi'].median():.1f}")
plt.title("AQI Value Distribution", fontsize=14); plt.xlabel("AQI"); plt.ylabel("Hours"); plt.legend(); plt.tight_layout()
plt.savefig("eda_5_distribution.png", dpi=150); plt.show()
print("✅ Saved eda_5_distribution.png")

print("\n🎉 EDA done! Save the 5 PNG files — paste them into your report.")