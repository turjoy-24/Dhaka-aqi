"""Dhaka PM2.5 dashboard. Run: streamlit run app.py"""
from datetime import date, timedelta

import pandas as pd
import streamlit as st

import pipeline as p

st.set_page_config(page_title="Dhaka PM2.5", layout="wide")
st.title("Dhaka Air Quality (PM2.5) Dashboard")
st.caption("Data source: Open-Meteo (CAMS model estimates, not ground sensors).")


@st.cache_data(ttl=6 * 3600)
def load_history():
    end = (date.today() - timedelta(days=3)).isoformat()
    return p.fetch_history("2022-09-01", end)


@st.cache_resource(ttl=6 * 3600)
def build_models(_hourly):
    return p.train_all(_hourly)


with st.spinner("Loading data and training the models..."):
    try:
        hourly = load_history()
        models = build_models(hourly)
    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.stop()

tab1, tab2, tab3, tab4 = st.tabs(
    ["History", "Model results", "Rain vs pollution", "Forecast"]
)

# ---------- Tab 1: history ----------
with tab1:
    daily = hourly["pm2_5"].resample("D").mean()
    start, end = st.slider(
        "Date range",
        min_value=daily.index.min().date(),
        max_value=daily.index.max().date(),
        value=(daily.index.max().date() - timedelta(days=365), daily.index.max().date()),
    )
    view = daily[str(start):str(end)]
    c1, c2, c3 = st.columns(3)
    c1.metric("Average PM2.5", f"{view.mean():.1f}")
    c2.metric("Worst day", f"{view.max():.1f}", str(view.idxmax().date()), delta_color="off")
    c3.metric("Cleanest day", f"{view.min():.1f}", str(view.idxmin().date()), delta_color="off")
    st.line_chart(view.rename("Daily PM2.5 (µg/m³)"))

    st.subheader("Average PM2.5 by month")
    by_month = daily.groupby(daily.index.month).mean().round(1)
    by_month.index.name = "month"
    st.bar_chart(by_month.rename("PM2.5"))

# ---------- Tab 2: model ----------
with tab2:
    st.write(
        "Time-based split (first 80% of days to train, last 20% to test). "
        "MAE in µg/m³, lower is better. Baseline = \"future = today\"."
    )
    rows = []
    for h in p.HORIZONS:
        m = models[h][3]
        rows.append(
            {
                "Days ahead": h,
                "Test days": m["test_days"],
                "Baseline MAE": m["baseline_mae"],
                "Model MAE": m["model_mae"],
                "Improvement %": (1 - m["model_mae"] / m["baseline_mae"]) * 100,
                "Winter baseline MAE": m["baseline_mae_winter"],
                "Winter model MAE": m["model_mae_winter"],
                "Range (low %)": m["range_low"] * 100,
                "Range (high %)": m["range_high"] * 100,
            }
        )
    st.dataframe(pd.DataFrame(rows).set_index("Days ahead").round(1))

    h = st.selectbox("Show details for forecast horizon (days ahead)", p.HORIZONS)
    _, test, pred, _, importance = models[h]

    st.subheader(f"Actual vs predicted, {h} day(s) ahead (test period)")
    st.line_chart(pd.DataFrame({"actual": test["target"], "predicted": pred}))

    st.subheader("Feature importance")
    st.bar_chart(importance)
    st.caption(
        "Backtest note: the model was given the *actual* weather of the target day, "
        "so these numbers are an upper limit. Real weather forecasts are less accurate, "
        "especially several days ahead."
    )

# ---------- Tab 3: rain ----------
with tab3:
    rt = p.rain_table(hourly)
    year = st.selectbox("Year", sorted(rt["year"].unique(), reverse=True))
    ry = rt[rt["year"] == year]
    overall = ry.groupby("rainy")["pm2_5"].agg(["mean", "count"]).round(1)
    overall.index = overall.index.map({True: "Rainy (>= 1 mm)", False: "Not rainy"})
    st.dataframe(overall)

    st.subheader("Same month, rainy vs not rainy")
    by_month = ry.groupby(["month", "rainy"])["pm2_5"].mean().unstack().round(1)
    by_month = by_month.rename(columns={True: "Rainy", False: "Not rainy"})
    st.dataframe(by_month)
    st.caption(
        "Compare within the same month: rain mostly falls in summer, when pollution "
        "is already lower, so the overall table alone can mislead."
    )

# ---------- Tab 4: forecast ----------
with tab4:
    st.write("Uses the last 10 days plus Open-Meteo's weather forecast.")
    try:
        fc = p.fetch_recent_forecast()
        today = pd.Timestamp.now(tz=p.TZ).tz_localize(None).normalize()

        cols = st.columns(len(p.HORIZONS))
        chart_rows = []
        today_pm = None
        for col, h in zip(cols, p.HORIZONS):
            model_h, _, _, m_h, _ = models[h]
            pm, today_pm = p.predict_ahead(model_h, fc, today, h)
            low = pm * (1 + m_h["range_low"])
            high = pm * (1 + m_h["range_high"])
            day = today + pd.Timedelta(days=h)
            col.metric(
                f"{day.strftime('%a %d %b')} (+{h} day)",
                f"{pm:.1f}",
                p.aqi_category(pm),
                delta_color="off",
            )
            col.caption(f"Expected range: {low:.0f} to {high:.0f} µg/m³")
            chart_rows.append({"date": day, "predicted": pm, "low": low, "high": high})

        st.metric("Today's PM2.5 (avg)", f"{today_pm:.1f}", p.aqi_category(today_pm), delta_color="off")

        chart = pd.DataFrame(chart_rows).set_index("date")
        chart.loc[today] = [today_pm, today_pm, today_pm]
        st.line_chart(chart.sort_index())

        st.caption(
            "Each range comes from that model's past errors on its test period; "
            "about 80% of past actual values fell inside such a range. "
            "Forecasts further ahead are less accurate (see the Model results tab). "
            "Learning project, not an official forecast or health advice."
        )
    except Exception as e:
        st.warning(f"Could not make a forecast right now: {e}")
