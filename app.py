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
def build_model(_hourly):
    d = p.make_training_data(_hourly)
    return p.train_and_evaluate(d)


with st.spinner("Loading data and training the model..."):
    try:
        hourly = load_history()
        model, test, pred, metrics, importance = build_model(hourly)
    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.stop()

tab1, tab2, tab3, tab4 = st.tabs(
    ["History", "Model results", "Rain vs pollution", "Tomorrow's forecast"]
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
        f"Time-based split: {metrics['train_days']} training days, "
        f"{metrics['test_days']} test days. Metric: MAE (lower is better)."
    )
    table = pd.DataFrame(
        {
            "All test days": [metrics["baseline_mae"], metrics["model_mae"]],
            "Winter (Dec-Mar)": [metrics["baseline_mae_winter"], metrics["model_mae_winter"]],
        },
        index=["Baseline (tomorrow = today)", "Random Forest"],
    ).round(1)
    st.dataframe(table)

    st.subheader("Actual vs predicted (test period)")
    chart = pd.DataFrame({"actual": test["target"], "predicted": pred})
    st.line_chart(chart)

    st.subheader("Feature importance")
    st.bar_chart(importance)
    st.caption(
        "Backtest note: the model was given tomorrow's *actual* weather, so these "
        "numbers are an upper limit. Real forecasts are less accurate."
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

# ---------- Tab 4: tomorrow ----------
with tab4:
    st.write("Uses the last 10 days plus Open-Meteo's weather forecast.")
    try:
        fc = p.fetch_recent_forecast()
        today = pd.Timestamp.now(tz=p.TZ).tz_localize(None).normalize()
        tomorrow_pm, today_pm = p.predict_tomorrow(model, fc, today)
        c1, c2 = st.columns(2)
        c1.metric("Today's PM2.5 (avg)", f"{today_pm:.1f}", p.aqi_category(today_pm), delta_color="off")
        c2.metric("Tomorrow's PM2.5 (predicted)", f"{tomorrow_pm:.1f}",
                  p.aqi_category(tomorrow_pm), delta_color="off")
        st.caption("Learning project, not an official forecast or health advice.")
    except Exception as e:
        st.warning(f"Could not make a forecast right now: {e}")
