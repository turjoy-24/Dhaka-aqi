# Dhaka Air Quality (PM2.5) Analysis and Next-Day Forecast

A learning project that collects hourly air quality and weather data for Dhaka, explores how weather relates to pollution, and builds a model that predicts **tomorrow's average PM2.5**. A Streamlit dashboard shows the results.

## Data

- **Air quality** (PM2.5, PM10): [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api)
- **Weather** (temperature, humidity, wind speed, precipitation): [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- Location: Dhaka (23.81 N, 90.41 E). Period: 1 Sep 2022 to 30 Sep 2026, hourly (35,784 rows).

**Important:** the air quality values are CAMS atmospheric *model estimates*, not readings from ground sensors. The API's archive starts in August 2022, which is why the project starts there. A natural next step is to repeat the work with real sensor data (for example OpenAQ or the US Embassy monitor in Dhaka).

## What I did, step by step

1. **Collected** hourly air quality and weather from two APIs and merged them on timestamp. There were no missing values in this model-based data.
2. **Explored one month (September 2026).** Daily PM2.5 ranged from about 10 to 64 µg/m³ and moved in waves rather than staying flat.
3. **Checked weather relationships** on those 30 days. PM2.5 correlated negatively with wind speed (-0.62) and rainfall (-0.55). Wind and rain are themselves correlated (0.61), so their effects cannot be separated from this alone, and 30 days is a small sample.
4. **Built a baseline:** "tomorrow = today". Any model has to beat it.
5. **Time-based split:** first 80% of days for training, last 20% for testing. No random shuffling, to avoid using the future to predict the past.
6. **Compared models** and added features step by step:

| Setup | Linear Regression | Random Forest | Baseline |
|---|---|---|---|
| Today's weather + PM2.5 | 8.2 | 8.4 | 8.7 |
| + lag features (1 day, 2 days, 7-day average) | 8.1 | 8.1 | 8.7 |
| + tomorrow's weather | 7.4 | 7.1 | 8.7 |

   MAE in µg/m³, lower is better. XGBoost scored 6.9 overall, which is within noise of Random Forest on a single test year.

7. **Checked where the model helps.** MAE by month showed larger errors in winter, but winter PM2.5 is also much higher. Relative to the baseline, the model helped most in winter: 9.7 vs 12.6 MAE for Dec-Mar.
8. **Feature importance:** today's PM2.5 accounts for about 78% of Random Forest importance; tomorrow's wind is the most useful weather feature (about 5%).
9. **Rain vs pollution:** compared rainy days (>= 1 mm) with non-rainy days, and also within the same month, because rain falls mostly in summer when pollution is already lower.

## Key findings

- "Tomorrow = today" is a strong baseline. Models beat it by about 18-21%, and by more in winter.
- Today's PM2.5 carries most of the signal; weather adds a smaller improvement.
- A simpler model was as good as a more complex one. XGBoost did not clearly beat Random Forest.

## Limitations

- Air quality data is model-based, not from sensors.
- The test set is about 300 days (one year), so differences of a few tenths in MAE are probably noise.
- In the backtest the model received tomorrow's **actual** weather. That is an upper limit; real forecasts are less accurate. The dashboard's live forecast tab uses Open-Meteo's weather forecast instead.
- Only one location, and no hyperparameter tuning beyond a single configuration.

## Run the dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

Files: `pipeline.py` (data, features, model) and `app.py` (dashboard with four tabs: history, model results, rain vs pollution, tomorrow's forecast).

## Possible next steps

- Use real sensor data (OpenAQ) and compare with the model-based data.
- Walk-forward validation across several years instead of one test year.
- Multi-step forecasts (3 days, 7 days).
- Add more cities.
