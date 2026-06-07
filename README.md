# Islamabad AQI Predictor

A fully automated, end-to-end Air Quality Index (AQI) prediction system for Islamabad, Pakistan. The system fetches real-time air quality data every hour, trains machine learning models daily, and serves a 3-day AQI forecast through an interactive web dashboard.

## Live Dashboard
**[View Live App](https://aqi-predictor-isb.streamlit.app/)**

## Project Overview
This project was built as part of the 10 Pearls Internship Program. It predicts the Air Quality Index for Islamabad for the next 3 days using a fully serverless ML pipeline.

## Architecture
OpenMeteo API (free, no key needed)
-->
GitHub Actions (runs every hour)
-->
MongoDB Atlas (Feature Store)
-->
Feature Engineering Pipeline
-->
Model Training (daily retraining)
-->
Streamlit Dashboard (live forecast)

## Tech Stack
| Component | Technology |
|---|---|
| Data Source | OpenMeteo API |
| Feature Store | MongoDB Atlas |
| Automation | GitHub Actions |
| ML Models | Scikit-learn + TensorFlow |
| Explainability | SHAP |
| Dashboard | Streamlit |
| Deployment | Streamlit Cloud |

## Features
- Real-time hourly AQI data collection for Islamabad
- 18 engineered features including lag features, rolling averages, time-based features
- 3 ML models trained and compared: Linear Regression, Random Forest, Neural Network
- Automated hourly data pipeline via GitHub Actions
- Automated daily model retraining via GitHub Actions
- Full pipeline logging stored in MongoDB
- Interactive dashboard with current AQI, 3-day forecast, and historical trend
- AQI hazard alerts when air quality is unhealthy

## Model Performance
| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 15.61 | 11.79 | 0.664 |
| Neural Network | 15.76 | 12.31 | 0.658 |
| Random Forest | 16.34 | 12.59 | 0.632 |

Best model: **Linear Regression**

## Data
- Source: OpenMeteo API (free, no API key required)
- Location: Islamabad, Pakistan (33.6844°N, 73.0479°E)
- Coverage: January 2025 — present (~12,000+ hourly records)
- Features collected: AQI, PM2.5, PM10, Temperature, Humidity, Wind Speed, Precipitation

## Automated Pipeline
- **Hourly:** GitHub Actions fetches latest AQI data and stores in MongoDB
- **Daily:** GitHub Actions retrains all 3 models and saves the best one
- All runs are logged to MongoDB `pipeline_logs` collection with timestamp, status, and environment