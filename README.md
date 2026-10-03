 Dhaka Air Quality (PM2.5) Analysis and Next-Day Forecast

Live dashboard: https://dhaka-aqi-qjtzqybfzwndapppzv227xz.streamlit.app

A learning project that collects hourly air quality and weather data for Dhaka, explores how weather relates to pollution, and builds a model that predicts tomorrow's average PM2.5. A Streamlit dashboard shows the results.

Data
Air quality (PM2.5, PM10): Open-Meteo Air Quality API
Weather (temperature, humidity, wind speed, precipitation): Open-Meteo Historical Weather API
Location: Dhaka (23.81 N, 90.41 E). Period: 1 Sep 2022 to 30 Sep 2026, hourly (35,784 rows).

Important: the air quality values are CAMS atmospheric model estimates, not readings from ground sensors. The API's archive starts in August 2022, which is why the project starts there. A natural next step is to repeat the work with real sensor data (for example OpenAQ or the US Embassy monitor in Dhaka).

What I did, step by step
Collected hourly air quality and weather from two APIs and merged them on timestamp. There were no missing values in this model-based data.
Explored one month (September 2026). Daily PM2.5 ranged from about 10 to 64 µg/m³ and moved in waves rather than staying flat.
Checked weather relationships on those 30 days. PM2.5 correlated negatively with wind speed (-0.62) and rainfall (-0.55). Wind and rain are themselves correlated (0.61), so their effects cannot be separated from this alone, and 30 days is a small sample.
Built a baseline: "tomorrow = today". Any model has to beat it.
Time-based split: first 80% of days for training, last 20% for testing. No random shuffling, to avoid using the future to predict the past.
Compared models and added features step by step:
Setup	Linear Regression	Random Forest	Baseline
Today's weather + PM2.5	8.2	8.4	8.7
+ lag features (1 day, 2 days, 7-day average)	8.1	8.1	8.7
+ tomorrow's weather	7.4	7.1	8.7
+ ## CAMS vs a ground sensor

I compared the Open-Meteo (CAMS) PM2.5 used in this project with the PM2.5 monitor "Dhaka" on OpenAQ (AirNow provider), using 840 days between Sep 2022 and Mar 2025 (days with at least 18 valid hours).
 CAMS vs ground sensors

The PM2.5 used in this project comes from Open-Meteo (CAMS model estimates). I compared it with three PM2.5 monitors on OpenAQ, using days with at least 18 valid hourly readings. Each row uses its own period, so averages are not comparable across rows.

| Sensor | Period | Common days | Daily correlation | Avg sensor | Avg CAMS | Bias (CAMS - sensor) |
|---|---|---|---|---|---|---|
| "Dhaka" (AirNow monitor) | Sep 2022 - Mar 2025 | 840 | 0.79 | 104 | 50 | -54 |
| Jahangirnagar University (AirGradient) | Oct 2024 - Sep 2026 | 336 | 0.83 | 122 | 58 | -64 |
| Uttara (AirGradient) | Dec 2025 - Sep 2026 | 294 | 0.87 | 93 | 47 | -46 |

All values in µg/m³. CAMS follows the ups and downs of all three sensors (correlation 0.79-0.87) but reports roughly half the level. The sensor/CAMS ratio ranges from about 1.3x to 2.7x depending on month and sensor, and is highest in Dec-Feb. The forecast errors in this README are measured against CAMS, not against ground sensors, and the AQI categories in the dashboard are based on CAMS values, so they likely understate real pollution.

Caveats: the sensors are in different parts of the city (and Jahangirnagar is in Savar), CAMS averages over a large grid cell, the two AirGradient devices are low-cost and were not calibrated here, and the Jahangirnagar data has gaps.
| Measure | Value |
|---|---|
| Daily correlation | 0.79 |
| Hourly correlation | 0.68 |
| Average sensor PM2.5 | 104 µg/m³ |
| Average CAMS PM2.5 | 50 µg/m³ |
| Bias (CAMS - sensor) | -54 µg/m³ |

CAMS follows the ups and downs of the sensor but reports roughly half the level (sensor/CAMS ratio about 1.6x to 2.7x depending on the month). The forecast errors above are measured against CAMS, not against the ground sensor, and the AQI categories in the dashboard are based on CAMS values, so they likely understate real pollution. This is a comparison with a single monitor in one part of the city, so part of the gap may come from CAMS averaging over a large grid cell.

MAE in µg/m³, lower is better. XGBoost scored 6.9 overall, which is within noise of Random Forest on a single test year.

Checked where the model helps. MAE by month showed larger errors in winter, but winter PM2.5 is also much higher. Relative to the baseline, the model helped most in winter: 9.7 vs 12.6 MAE for Dec-Mar.
Feature importance: today's PM2.5 accounts for about 78% of Random Forest importance; tomorrow's wind is the most useful weather feature (about 5%).
Rain vs pollution: compared rainy days (>= 1 mm) with non-rainy days, and also within the same month, because rain falls mostly in summer when pollution is already lower.
Key findings
"Tomorrow = today" is a strong baseline. On the single split, the models beat it by about 15-21%, and by more in winter.
Today's PM2.5 carries most of the signal; weather adds a smaller improvement.
A simpler model was as good as a more complex one. XGBoost did not clearly beat Random Forest.
Walk-forward validation

To check the result does not depend on one test year, I trained on all earlier data and tested on the next year, three times (MAE in µg/m³, lower is better):

Test year	Train days	Test days	Baseline	Linear	Random Forest
2024	480	366	9.0	8.3	8.9
2025	846	365	10.0	8.9	9.4
2026 (Jan-Sep)	1211	272	8.7	7.3	6.9
Average			9.2	8.2	8.4

Both models beat the baseline in every fold. On average the improvement is about 9-11%, smaller than the single-split numbers above. Linear Regression was better on average; Random Forest was better only in the last fold, which had the most training data. The 2026 fold has no Oct-Dec data.

Prediction range

The dashboard also shows an expected range for tomorrow, built from the 10th and 90th percentile of the model's past relative errors (actual / predicted - 1) on the test period. In the original test period this was roughly -30% to +24%. The live range on the dashboard is recalculated whenever the model retrains, so the exact percentages can differ slightly.

To check the range honestly, I built it from 2025 errors and tested it on 2026: about 82% of actual values fell inside it (89% in Jan-Mar, 79% in Apr-Sep). The range is wide, which reflects how hard next-day PM2.5 is to predict, and it is not a guarantee.

Limitations
Air quality data is model-based, not from sensors.
The single-split test set is about 300 days (one year), so differences of a few tenths in MAE are probably noise.
In the backtest the model received tomorrow's actual weather. That is an upper limit; real forecasts are less accurate. The dashboard's live forecast tab uses Open-Meteo's weather forecast instead.
Only one location, and no hyperparameter tuning beyond a single configuration.
Run the dashboard
bash
pip install -r requirements.txt
streamlit run app.py

Files: pipeline.py (data, features, model) and app.py (dashboard with four tabs: history, model results, rain vs pollution, tomorrow's forecast).

Possible next steps
Use real sensor data (OpenAQ) and compare with the model-based data.
Multi-step forecasts (3 days, 7 days).
Add more cities.
