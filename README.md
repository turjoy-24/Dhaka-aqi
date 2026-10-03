 Dhaka Air Quality (PM2.5): Analysis and 1/3/7-Day Forecast

**Live dashboard:** https://dhaka-aqi-qjtzqybfzwndapppzv227xz.streamlit.app/

A data science project on air pollution in Dhaka. It collects four years of hourly PM2.5 and weather data, studies how weather relates to pollution, forecasts daily PM2.5 for the next 1, 3 and 7 days with an expected range and an explanation of each prediction, and checks the data source against real ground sensors.

![Project pipeline](images/pipeline.png)

## Highlights

- **Forecasts beat the "tomorrow = today" baseline in every validation year**, by about 9-11% on average across three yearly walk-forward folds (up to about 21% in the latest fold, and by more in winter).
- **Today's PM2.5 carries most of the signal** (about 78% of Random Forest importance for the 1-day model). Next-day wind is the most useful weather feature.
- **The PM2.5 source used here (CAMS model estimates) reports roughly half the level measured by three ground sensors in Dhaka**, although it follows their ups and downs closely (daily correlation 0.79-0.87). Forecast errors below are therefore measured against CAMS, not against real air.
- A simple model was as good as a complex one: Linear Regression, Random Forest and XGBoost ended up within noise of each other.

## Dashboard

The Streamlit app has four tabs:

| Tab | What it shows |
|---|---|
| **History** | Daily PM2.5 for any date range, worst and cleanest day, average by month |
| **Model results** | Error of the 1, 3 and 7 day models against the baseline, actual vs predicted chart, feature importance |
| **Rain vs pollution** | Rainy vs non-rainy days, overall and within the same month |
| **Forecast** | PM2.5 for tomorrow, in 3 days and in 7 days with an expected range, plus a SHAP explanation of each forecast |

## Data

| Source | Used for |
|---|---|
| [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) | PM2.5 and PM10 (CAMS atmospheric model estimates) |
| [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) and Forecast API | Temperature, humidity, wind speed, precipitation |
| [OpenAQ](https://openaq.org) | Ground sensor PM2.5, used only to check CAMS (see below) |

Location: Dhaka (23.81 N, 90.41 E). Period: 1 Sep 2022 to 30 Sep 2026, 35,784 hourly rows. The air quality archive starts in August 2022, which is why the project starts there. Because the data is model-based, it has no missing values and needs little cleaning.

## Exploring the data (September 2026)

![Daily PM2.5 in September 2026](images/sep2026_daily_pm25.png)

Daily PM2.5 ranged from about 10 to 64 µg/m³ and moved in waves rather than staying flat, with peaks around 8, 19 and 28 September. On these 30 days PM2.5 correlated negatively with wind speed (-0.62) and rainfall (-0.55). Wind and rain are themselves correlated (0.61), so their effects cannot be separated, and 30 days is a small sample.

Category colours use simplified US-EPA-style PM2.5 breakpoints (12 / 35 / 55 µg/m³).

## Modelling

**Task:** predict the daily mean PM2.5 1, 3 and 7 days ahead.

**Baseline:** "the future equals today". Every model has to beat it.

**Features:** today's PM2.5, PM10 and weather; PM2.5 yesterday and two days ago; the 7-day mean; month; and the wind, rain and humidity of the **target day**. One Random Forest is trained per horizon.

**Validation:** a time-based split (first 80% of days to train, last 20% to test, no shuffling, so the future never predicts the past). The test set is 298 days, roughly Dec 2025 to Sep 2026, with an average PM2.5 of 46.8 µg/m³.

### Results for the 1-day forecast

MAE in µg/m³ on the test set (lower is better). Features were added step by step:

| Setup | Linear Regression | Random Forest | Baseline |
|---|---|---|---|
| Today's PM2.5 and weather | 8.2 | 8.4 | 8.7 |
| + lag features (1 day, 2 days, 7-day mean) | 8.1 | 8.1 | 8.7 |
| + weather of the target day | 7.4 | 7.1 | 8.7 |

XGBoost scored 6.9 overall, which is within noise of Random Forest on one test year (in winter it scored 9.9 vs 9.7 for Random Forest).

![Feature importance](images/feature_importance.png)

### Where the model helps

![Error by month](images/error_by_month.png)

Winter has the largest errors in absolute terms, but winter pollution is also much higher. Compared with the baseline, the model helped most in winter: for Dec-Mar, Random Forest scored 9.7 MAE against 12.6 for the baseline (about 23% better). In April and December the baseline was slightly better, and in May and June the two were equal.

### Walk-forward validation

To check the result does not depend on one test year, I trained on all earlier data and tested on the next year, three times:

![Walk-forward validation](images/walk_forward_mae.png)

| Test year | Train days | Test days | Baseline | Linear | Random Forest |
|---|---|---|---|---|---|
| 2024 | 480 | 366 | 9.0 | 8.3 | 8.9 |
| 2025 | 846 | 365 | 10.0 | 8.9 | 9.4 |
| 2026 (Jan-Sep) | 1211 | 272 | 8.7 | 7.3 | 6.9 |
| **Average** | | | **9.2** | **8.2** | **8.4** |

Both models beat the baseline in every fold, but by less than the single split suggested: about 9-11% on average. Linear Regression was better on average; Random Forest won only in the last fold, which had the most training data. The 2026 fold has no Oct-Dec data.

## Forecast horizons and expected range

The dashboard forecasts 1, 3 and 7 days ahead. Longer horizons are harder, and the **Model results** tab reports the measured error of each horizon.

Each forecast comes with an **expected range**: the 10th and 90th percentile of that model's past relative errors (actual / predicted - 1). For the 1-day model, a range built from 2025 errors (about -30% to +24% around the prediction) was tested on 2026 and contained **82%** of the actual values (89% in Jan-Mar, 79% in Apr-Sep). Using relative instead of fixed-size ranges mattered: a fixed range covered 90% of values in summer but only 73% in winter.

In the backtest the models receive the **actual** weather of the target day. Real weather forecasts are less accurate, especially several days ahead, so the backtest errors are an upper limit of performance and the 7-day forecast should be read as a rough indication.

## Explaining predictions

The Forecast tab uses [SHAP](https://github.com/shap/shap) to show how much each feature pushed one prediction up or down from the model's average prediction. This describes how the model behaves, not what causes pollution, and related features (today's PM2.5, yesterday's, the 7-day mean) share credit.

## CAMS vs ground sensors

I compared the CAMS PM2.5 used in this project with three PM2.5 monitors on OpenAQ, using days with at least 18 valid hourly readings.

![CAMS vs ground sensors](images/cams_vs_sensors.png)

| Sensor | Period | Common days | Daily correlation | Avg sensor | Avg CAMS | Bias (CAMS - sensor) |
|---|---|---|---|---|---|---|
| "Dhaka" (AirNow provider) | Sep 2022 - Mar 2025 | 840 | 0.79 | 104 | 50 | -54 |
| Jahangirnagar University (AirGradient) | Oct 2024 - Sep 2026 | 336 | 0.83 | 122 | 58 | -64 |
| Uttara, RAJUK (AirGradient) | Dec 2025 - Sep 2026 | 294 | 0.87 | 93 | 47 | -46 |

All values in µg/m³. Each row uses its own period, so averages are not comparable across rows. CAMS captures the rises and falls of all three sensors but reports roughly half the level. The sensor/CAMS ratio ranges from about 1.3x to 2.7x depending on month and sensor and tends to be highest around December and January (about 2.4x to 2.7x). Because the AQI categories shown in the dashboard are based on CAMS values, **they likely understate real pollution**.

Caveats: the sensors are in different parts of the city (Jahangirnagar is in Savar) while CAMS averages over a large grid cell; the two AirGradient devices are low-cost and were not calibrated here; the Jahangirnagar data has gaps.

## Limitations

- The PM2.5 used for training and forecasting is a model estimate that runs at about half the level of three ground sensors. Forecasts are for CAMS values, not for real air.
- Differences of a few tenths in MAE between models are probably noise. Each validation fold has only 272-366 test days.
- Backtests use actual weather for the target day, so they are optimistic, especially for the 7-day horizon.
- Only one location, one model family tuned with a single configuration, and no hyperparameter search.
- Not an official forecast and not health advice.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

```
dhaka-aqi/
├── app.py            # Streamlit dashboard
├── pipeline.py       # data collection, features, models, SHAP
├── requirements.txt
├── images/           # charts used in this README
└── README.md
```

## Possible next steps

- Correct CAMS using the ground sensors and forecast the corrected values.
- Use real weather forecasts in the backtest instead of actual weather.
- Add more cities (Chattogram, Sylhet) and compare them.
- Add a classification view (probability of a high-pollution day) and hyperparameter search.

