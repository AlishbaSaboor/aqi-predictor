import shap, joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pymongo import MongoClient
import os, sys
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.logger import log_run

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

FEATURES = [
    "hour","day","month","weekday","is_weekend",
    "aqi_lag_1","aqi_lag_3","aqi_lag_6","aqi_lag_24","aqi_lag_48",
    "aqi_rolling_3h","aqi_rolling_24h","aqi_change_rate",
    "temperature","humidity","wind_speed","pm2_5","pm10",
]

print("🔗 Loading data ...")
client  = MongoClient(os.getenv("MONGO_URI"))
records = list(client[os.getenv("MONGO_DB")]["engineered_features"].find({}, {"_id": 0}))
client.close()
df = pd.DataFrame(records).dropna(subset=FEATURES + ["target_24h"])
X  = df[FEATURES].values.astype(float)
print(f"✅ Loaded {len(df)} rows")

try:
    rf          = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    sample      = X[:500]
    explainer   = shap.TreeExplainer(rf)
    shap_vals   = explainer.shap_values(sample)

    # Chart 1: Feature importance bar
    plt.figure(figsize=(10,7))
    shap.summary_plot(shap_vals, sample, feature_names=FEATURES, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance — Which features drive AQI predictions?")
    plt.tight_layout()
    p1 = os.path.join(MODELS_DIR, "shap_importance.png")
    plt.savefig(p1, dpi=150, bbox_inches="tight"); plt.show()
    print(f"✅ Saved {p1}")

    # Chart 2: SHAP beeswarm (shows direction)
    plt.figure(figsize=(10,7))
    shap.summary_plot(shap_vals, sample, feature_names=FEATURES, show=False)
    plt.title("SHAP Values — Direction and magnitude per feature")
    plt.tight_layout()
    p2 = os.path.join(MODELS_DIR, "shap_beeswarm.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight"); plt.show()
    print(f"✅ Saved {p2}")

    log_run("shap_analysis", "success", extra_info={"samples": 500})
    print("🎉 SHAP done!")

except Exception as err:
    log_run("shap_analysis", "failed", error_message=str(err))
    print(f"❌ SHAP failed: {err}")
    raise