import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
import joblib, json, os, sys
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.logger import log_run, log_training

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURES = [
    "hour","day","month","weekday","is_weekend",
    "aqi_lag_1","aqi_lag_3","aqi_lag_6","aqi_lag_24","aqi_lag_48",
    "aqi_rolling_3h","aqi_rolling_24h","aqi_change_rate",
    "temperature","humidity","wind_speed","pm2_5","pm10",
]
TARGET = "target_24h"

def load_data():
    print("Loading training data from MongoDB ...")
    client  = MongoClient(os.getenv("MONGO_URI"))
    records = list(client[os.getenv("MONGO_DB")]["engineered_features"].find({}, {"_id": 0}))
    client.close()
    df = pd.DataFrame(records).dropna(subset=FEATURES + [TARGET])
    X  = df[FEATURES].values.astype(float)
    y  = df[TARGET].values.astype(float)
    print(f"Loaded {len(df)} samples with {len(FEATURES)} features")
    return X, y, len(df)

def evaluate(name, y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    print(f"  {name:22s}  RMSE={rmse:7.2f}  MAE={mae:7.2f}  R²={r2:.4f}")
    return {"rmse": round(rmse,4), "mae": round(mae,4), "r2": round(r2,4)}

def train_all():
    try:
        X, y, total = load_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        scaler    = StandardScaler()
        Xtr_s     = scaler.fit_transform(X_train)
        Xte_s     = scaler.transform(X_test)

        results = {}
        print(f"\n{'='*60}")
        print(f"{'Model':22s}  {'RMSE':>7}  {'MAE':>7}  {'R²':>7}")
        print(f"{'─'*60}")

        #Linear Regression
        lr = LinearRegression()
        lr.fit(Xtr_s, y_train)
        results["LinearRegression"] = evaluate("LinearRegression", y_test, lr.predict(Xte_s))

        #Random Forest
        print("  [Training Random Forest... ~1 min]")
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        results["RandomForest"] = evaluate("RandomForest", y_test, rf.predict(X_test))

        #Neural Network
        print("  [Training Neural Network... ~3 min]")
        tf.random.set_seed(42)
        nn = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu", input_shape=(len(FEATURES),)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64,  activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32,  activation="relu"),
            tf.keras.layers.Dense(1),
        ])
        nn.compile(optimizer="adam", loss="mse", metrics=["mae"])
        nn.fit(Xtr_s, y_train, epochs=50, batch_size=64,
               validation_split=0.1, verbose=0,
               callbacks=[tf.keras.callbacks.EarlyStopping(
                   patience=5, restore_best_weights=True)])
        results["NeuralNetwork"] = evaluate("NeuralNetwork", y_test, nn.predict(Xte_s, verbose=0).flatten())

        print(f"{'='*60}")

        best = min(results, key=lambda k: results[k]["rmse"])
        print(f"\nBest model: {best}  (RMSE = {results[best]['rmse']})")

        joblib.dump(lr,     os.path.join(MODELS_DIR, "linear_regression.pkl"))
        joblib.dump(rf,     os.path.join(MODELS_DIR, "random_forest.pkl"))
        joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
        nn.save(os.path.join(MODELS_DIR, "neural_network.keras"))

        with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
            json.dump(results, f, indent=2)
        with open(os.path.join(MODELS_DIR, "best_model.txt"), "w") as f:
            f.write(best)

        print("All models saved to /models/")

        log_training(
            best_model=best,
            all_metrics=results,
            train_rows=len(X_train),
            test_rows=len(X_test)
        )
        log_run(
            script="training",
            status="success",
            extra_info={
                "best_model": best,
                "best_rmse":  results[best]["rmse"],
                "best_r2":    results[best]["r2"],
                "total_rows": total,
            }
        )
        print("Training complete!")

    except Exception as err:
        log_run("training", "failed", error_message=str(err))
        print(f"raining failed: {err}")
        raise

if __name__ == "__main__":
    train_all()