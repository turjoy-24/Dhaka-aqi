"""Data + model code for the Dhaka PM2.5 project (no Streamlit in here)."""
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor

LAT, LON = 23.81, 90.41
TZ = "Asia/Dhaka"

AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

AIR_VARS = ["pm2_5", "pm10"]
WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"]

HORIZONS = (1, 3, 7)  # days ahead

BASE_FEATURES = [
    "pm2_5", "lag_1", "lag_2", "roll_7", "pm10",
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation",
    "month",
]


def feature_names(h):
    """Features for the model that predicts h days ahead.
    Includes the weather on the target day (wind, rain, humidity)."""
    return BASE_FEATURES + [f"wind_d{h}", f"rain_d{h}", f"humid_d{h}"]


# ---------- 1. Data collection ----------
def _get_hourly(url, variables, **extra):
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": ",".join(variables),
        "timezone": TZ,
        **extra,
    }
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def fetch_history(start, end):
    """Hourly air quality + weather between two dates (YYYY-MM-DD)."""
    air = _get_hourly(AIR_URL, AIR_VARS, start_date=start, end_date=end)
    weather = _get_hourly(WEATHER_ARCHIVE_URL, WEATHER_VARS, start_date=start, end_date=end)
    return air.join(weather, how="inner")


def fetch_recent_forecast():
    """Last 10 days of air quality + weather, plus weather forecast for the next ~9 days."""
    air = _get_hourly(AIR_URL, AIR_VARS, past_days=10, forecast_days=2)
    weather = _get_hourly(WEATHER_FORECAST_URL, WEATHER_VARS, past_days=10, forecast_days=9)
    # weather goes further into the future than air quality, so keep all weather rows
    return weather.join(air, how="left")


# ---------- 2. Features ----------
def make_features(hourly):
    """Hourly -> one row per day, with lag / rolling / future-weather features."""
    d = hourly.resample("D").mean()
    d["month"] = d.index.month
    d["lag_1"] = d["pm2_5"].shift(1)
    d["lag_2"] = d["pm2_5"].shift(2)
    d["roll_7"] = d["pm2_5"].rolling(7).mean()
    for h in HORIZONS:
        d[f"wind_d{h}"] = d["wind_speed_10m"].shift(-h)
        d[f"rain_d{h}"] = d["precipitation"].shift(-h)
        d[f"humid_d{h}"] = d["relative_humidity_2m"].shift(-h)
    return d


def make_training_data(hourly, h):
    d = make_features(hourly)
    d["target"] = d["pm2_5"].shift(-h)  # PM2.5 h days from now
    return d.dropna(subset=feature_names(h) + ["target"])


# ---------- 3. Model ----------
def train_and_evaluate(d, h, train_frac=0.8):
    """Time-based split (no shuffling), baseline vs Random Forest, for horizon h."""
    names = feature_names(h)
    split = int(len(d) * train_frac)
    train, test = d.iloc[:split], d.iloc[split:]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(train[names], train["target"])
    pred = pd.Series(model.predict(test[names]), index=test.index)

    winter = test.index.month.isin([12, 1, 2, 3])
    base_err = (test["target"] - test["pm2_5"]).abs()  # "future = today"
    model_err = (test["target"] - pred).abs()

    # Prediction range: 10th and 90th percentile of the relative error
    # (actual / predicted - 1) on the test period.
    rel = test["target"] / pred - 1
    range_low, range_high = rel.quantile(0.1), rel.quantile(0.9)

    metrics = {
        "horizon": h,
        "range_low": float(range_low),
        "range_high": float(range_high),
        "train_days": len(train),
        "test_days": len(test),
        "baseline_mae": float(base_err.mean()),
        "model_mae": float(model_err.mean()),
        "baseline_mae_winter": float(base_err[winter].mean()),
        "model_mae_winter": float(model_err[winter].mean()),
    }
    importance = pd.Series(model.feature_importances_, index=names).sort_values(ascending=False)
    return model, test, pred, metrics, importance


def train_all(hourly):
    """One model per horizon: {1: (...), 3: (...), 7: (...)}."""
    return {h: train_and_evaluate(make_training_data(hourly, h), h) for h in HORIZONS}


def predict_ahead(model, forecast_hourly, today, h):
    """Predict PM2.5 for `today + h days`. Returns (prediction, today's PM2.5)."""
    d = make_features(forecast_hourly)
    if today not in d.index:
        raise ValueError("Today's row is missing from the forecast data.")
    row = d.loc[[today], feature_names(h)]
    if row.isna().any(axis=None):
        missing = list(row.columns[row.isna().any()])
        raise ValueError(f"Not enough data to build today's features: {missing}")
    return float(model.predict(row)[0]), float(row["pm2_5"].iloc[0])


# ---------- 4. Rain vs pollution ----------
def rain_table(hourly, rain_mm=1.0):
    daily = hourly.resample("D").agg({"pm2_5": "mean", "precipitation": "sum"})
    daily["rainy"] = daily["precipitation"] >= rain_mm
    daily["year"] = daily.index.year
    daily["month"] = daily.index.month
    return daily


# ---------- 5. Helper ----------
def aqi_category(pm25):
    if pm25 < 12:
        return "Good"
    if pm25 < 35:
        return "Moderate"
    if pm25 < 55:
        return "Unhealthy for sensitive groups"
    if pm25 < 150:
        return "Unhealthy"
    return "Very unhealthy"
