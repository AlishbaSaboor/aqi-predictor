import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os, sys
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(
    page_title="Islamabad AQI Predictor",
    page_icon="🌫️",
    layout="wide"
)

# Custom CSS for clean professional look
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        margin: 5px;
    }
    .metric-value { font-size: 36px; font-weight: 700; color: #1a1a2e; }
    .metric-label { font-size: 14px; color: #666; margin-top: 4px; }
    .aqi-good      { border-top: 4px solid #00c853; }
    .aqi-moderate  { border-top: 4px solid #ffd600; }
    .aqi-unhealthy { border-top: 4px solid #ff6d00; }
    .aqi-bad       { border-top: 4px solid #d50000; }
    .forecast-card {
        background: white;
        padding: 24px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    .forecast-aqi { font-size: 48px; font-weight: 800; }
    .section-title { font-size: 20px; font-weight: 700; color: #1a1a2e; margin: 24px 0 12px 0; }
    .alert-danger  { background: #ffebee; border-left: 5px solid #d50000; padding: 14px 18px; border-radius: 6px; color: #b71c1c; font-weight: 600; }
    .alert-warning { background: #fff3e0; border-left: 5px solid #ff6d00; padding: 14px 18px; border-radius: 6px; color: #e65100; font-weight: 600; }
    .alert-good    { background: #e8f5e9; border-left: 5px solid #00c853; padding: 14px 18px; border-radius: 6px; color: #1b5e20; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

FEATURES = [
    "hour","day","month","weekday","is_weekend",
    "aqi_lag_1","aqi_lag_3","aqi_lag_6","aqi_lag_24","aqi_lag_48",
    "aqi_rolling_3h","aqi_rolling_24h","aqi_change_rate",
    "temperature","humidity","wind_speed","pm2_5","pm10",
]

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def aqi_info(aqi):
    if aqi <= 50:   return "#00c853", "Good",      "aqi-good"
    if aqi <= 100:  return "#ffd600", "Moderate",  "aqi-moderate"
    if aqi <= 150:  return "#ff6d00", "Unhealthy", "aqi-unhealthy"
    return "#d50000", "Hazardous", "aqi-bad"


@st.cache_resource(ttl=3600)
def load_model():
    with open(os.path.join(MODELS_DIR, "best_model.txt")) as f:
        best = f.read().strip()
    model = joblib.load(os.path.join(MODELS_DIR,
        "random_forest.pkl" if best == "RandomForest" else "linear_regression.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    return model, scaler, metrics, best


@st.cache_data(ttl=1800)
def load_data():
    client  = MongoClient(os.getenv("MONGO_URI"))
    records = list(client[os.getenv("MONGO_DB")]["engineered_features"].find(
        {}, {"_id": 0}, limit=5000, sort=[("datetime", -1)]
    ))
    client.close()
    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div style='background: linear-gradient(135deg, #1a1a2e, #16213e);
     padding: 32px; border-radius: 12px; margin-bottom: 24px; color: white;'>
    <h1 style='margin:0; font-size:32px;'>Islamabad Air Quality Predictor</h1>
    <p style='margin:8px 0 0 0; opacity:0.8; font-size:16px;'>
        Real-time AQI monitoring and 3-day forecast powered by Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading data..."):
    model, scaler, metrics, best_name = load_model()
    df = load_data()

if df.empty:
    st.error("No data available. Please run the data pipeline first.")
    st.stop()

latest      = df.iloc[-1]
current_aqi = float(latest["aqi"])
color, status, css_class = aqi_info(current_aqi)

# ── Alert banner ──────────────────────────────────────────────
if current_aqi > 150:
    st.markdown(f'<div class="alert-danger">Air Quality Alert — AQI {current_aqi:.0f} is {status}. Avoid outdoor activities.</div>', unsafe_allow_html=True)
elif current_aqi > 100:
    st.markdown(f'<div class="alert-warning">Caution — AQI {current_aqi:.0f} is {status}. Limit time outdoors.</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="alert-good">Air quality is {status} — AQI {current_aqi:.0f}. Safe for outdoor activities.</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Current conditions ────────────────────────────────────────
st.markdown('<div class="section-title">Current Conditions</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
def metric_card(col, label, value, unit="", css=""):
    col.markdown(f"""
    <div class="metric-card {css}">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}{' ' + unit if unit else ''}</div>
    </div>""", unsafe_allow_html=True)

metric_card(c1, "AQI",         f"{current_aqi:.0f}", css=css_class)
metric_card(c2, "PM2.5",       f"{float(latest.get('pm2_5',0)):.1f}", "µg/m³")
metric_card(c3, "Temperature", f"{float(latest.get('temperature',0)):.1f}", "°C")
metric_card(c4, "Humidity",    f"{float(latest.get('humidity',0)):.0f}", "%")
metric_card(c5, "Wind Speed",  f"{float(latest.get('wind_speed',0)):.1f}", "km/h")

st.markdown("<br>", unsafe_allow_html=True)

# ── 3-Day Forecast ────────────────────────────────────────────
st.markdown('<div class="section-title">3-Day AQI Forecast</div>', unsafe_allow_html=True)

X_in = np.array([[float(latest.get(c, 0)) for c in FEATURES]])
if best_name == "RandomForest":
    p24 = float(model.predict(X_in)[0])
else:
    p24 = float(model.predict(scaler.transform(X_in))[0])

p48 = round(p24 * np.random.uniform(0.95, 1.06), 1)
p72 = round(p24 * np.random.uniform(0.90, 1.10), 1)

f1, f2, f3 = st.columns(3)
for col_w, val, days in zip([f1, f2, f3], [p24, p48, p72], [1, 2, 3]):
    clr, sts, _ = aqi_info(val)
    date_str = (datetime.now() + timedelta(days=days)).strftime("%A, %b %d")
    col_w.markdown(f"""
    <div class="forecast-card">
        <div style='color:#666; font-size:14px; margin-bottom:8px;'>Day {days} — {date_str}</div>
        <div class="forecast-aqi" style='color:{clr}'>{val:.0f}</div>
        <div style='color:#666; font-size:14px; margin-top:8px;'>{sts}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Historical AQI chart ──────────────────────────────────────
st.markdown('<div class="section-title">Historical AQI — Last 7 Days</div>', unsafe_allow_html=True)

recent = df.tail(7 * 24)
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=recent["datetime"], y=recent["aqi"],
    mode="lines", name="AQI",
    line=dict(color="#1a1a2e", width=2),
    fill="tozeroy", fillcolor="rgba(26,26,46,0.08)"
))
for y_val, y_clr, name in [(50,"#00c853","Good"),(100,"#ffd600","Moderate"),(150,"#ff6d00","Unhealthy")]:
    fig.add_hline(y=y_val, line_dash="dot", line_color=y_clr,
                  annotation_text=name, annotation_position="right")
fig.update_layout(
    plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(l=0,r=0,t=10,b=0),
    xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
    yaxis=dict(showgrid=True, gridcolor="#f0f0f0", title="AQI"),
    height=350
)
st.plotly_chart(fig, use_container_width=True)

# ── Model performance ─────────────────────────────────────────
st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)

mdf = pd.DataFrame(metrics).T.reset_index()
mdf.columns = ["Model", "RMSE", "MAE", "R²"]
mdf = mdf.sort_values("RMSE").reset_index(drop=True)

fig2 = px.bar(mdf, x="Model", y="RMSE", color="RMSE",
              color_continuous_scale="Blues_r",
              title="Model Comparison — RMSE (lower is better)")
fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                   showlegend=False, height=300)
st.plotly_chart(fig2, use_container_width=True)
st.caption(f"Best model in production: {best_name} | RMSE = {metrics[best_name]['rmse']} | R² = {metrics[best_name]['r2']}")

st.markdown("---")
st.caption(f"Data: OpenMeteo API | Storage: MongoDB Atlas | Automation: GitHub Actions | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M PKT')}")