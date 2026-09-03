"""
app.py

Chennai Public Transport
FDS Exploratory & Predictive Analytics Dashboard

IMPORTANT:
Prediction is statistical/historical.
NO MACHINE LEARNING is used.
"""

import sys
import pathlib

sys.path.insert(
    0,
    str(pathlib.Path(__file__).resolve().parent)
)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.utils import RAW_CSV, CLEANED_CSV, ensure_dirs
from src.data_processing import (
    load_raw_data,
    clean_data,
    save_cleaned
)

from src.historical_predictor import (
    predict_trip,
    what_if_analysis,
    best_time_to_travel,
    get_time_period
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Chennai Public Transport Analytics",
    page_icon="🚌",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #123b5e;
}

.subtitle {
    font-size: 17px;
    color: #5c6b73;
}

.card {
    padding: 20px;
    border-radius: 16px;
    background: #ffffff;
    border: 1px solid #e5e9ed;
    margin-bottom: 15px;
}

.result-card {
    padding: 25px;
    border-radius: 18px;
    background: #f5f8fa;
    border: 1px solid #dce3e8;
}

.big-number {
    font-size: 36px;
    font-weight: 800;
}

.small-label {
    color: #687780;
    font-size: 14px;
}

.route-box {
    padding: 18px;
    border-radius: 14px;
    background: #eef6fb;
    border-left: 5px solid #2878a7;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def get_data():

    ensure_dirs()

    if CLEANED_CSV.exists():

        df = pd.read_csv(
            CLEANED_CSV,
            low_memory=False
        )

        if "Date" in df.columns:

            df["Date"] = pd.to_datetime(
                df["Date"],
                errors="coerce"
            )

        return df

    raw = load_raw_data()

    if raw.empty:
        return raw

    cleaned = clean_data(raw)

    save_cleaned(cleaned)

    return cleaned


df = get_data()


# ============================================================
# CHECK DATA
# ============================================================

if df.empty:

    st.error(
        "Dataset not found. Please check the data folder."
    )

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🚌 Chennai Public Transport</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Exploratory Data Analysis & Historical Predictive Analytics'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "📌 This project does NOT use Machine Learning. "
    "Predictions are generated using historical statistical patterns, "
    "conditional averages, percentiles and similarity-based analysis."
)


# ============================================================
# DATASET INTELLIGENCE
# ============================================================

st.header("📊 Dataset Intelligence")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Records",
    f"{len(df):,}"
)

c2.metric(
    "Features",
    f"{len(df.columns):,}"
)

c3.metric(
    "Missing Values",
    f"{int(df.isnull().sum().sum()):,}"
)

c4.metric(
    "Duplicate Rows",
    f"{int(df.duplicated().sum()):,}"
)


# ============================================================
# EDA
# ============================================================

st.header("🔎 Exploratory Data Analysis")

eda_tab1, eda_tab2, eda_tab3 = st.tabs(
    [
        "📈 Delay Distribution",
        "⏰ Time Patterns",
        "🌦️ Weather & Traffic"
    ]
)


# ------------------------------------------------------------
# TAB 1
# ------------------------------------------------------------

with eda_tab1:

    if "Delay_Minutes" in df.columns:

        fig = px.histogram(
            df,
            x="Delay_Minutes",
            nbins=30,
            title="Distribution of Transport Delay"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        fig2 = px.box(
            df,
            y="Delay_Minutes",
            title="Delay Outlier Analysis"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )


# ------------------------------------------------------------
# TAB 2
# ------------------------------------------------------------

with eda_tab2:

    if (
        "Hour" in df.columns
        and "Delay_Minutes" in df.columns
    ):

        hourly = (
            df.groupby("Hour")["Delay_Minutes"]
            .mean()
            .reset_index()
        )

        fig = px.line(
            hourly,
            x="Hour",
            y="Delay_Minutes",
            markers=True,
            title="Average Delay by Hour"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ------------------------------------------------------------
# TAB 3
# ------------------------------------------------------------

with eda_tab3:

    col1, col2 = st.columns(2)

    with col1:

        if (
            "Weather" in df.columns
            and "Delay_Minutes" in df.columns
        ):

            fig = px.box(
                df,
                x="Weather",
                y="Delay_Minutes",
                title="Delay by Weather"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with col2:

        if (
            "Traffic_Level" in df.columns
            and "Delay_Minutes" in df.columns
        ):

            traffic = (
                df.groupby("Traffic_Level")
                ["Delay_Minutes"]
                .mean()
                .reset_index()
            )

            fig = px.bar(
                traffic,
                x="Traffic_Level",
                y="Delay_Minutes",
                title="Average Delay by Traffic Level"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# SMART TRIP PREDICTION
# ============================================================

st.header("🧭 Smart Historical Trip Prediction")

st.write(
    "Choose your actual starting point and destination. "
    "The system analyses historical trips matching your "
    "route, starting point, day, time and conditions."
)


# ============================================================
# ROUTES
# ============================================================

routes = sorted(
    df["Route_Name"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

if not routes:

    st.error("No routes available.")

    st.stop()


route = st.selectbox(
    "🚌 Select Route",
    routes
)


route_df = df[
    df["Route_Name"].astype(str) == str(route)
].copy()


# ============================================================
# STARTING POINT
# ============================================================

if "Representative_Stop" in route_df.columns:

    starts = sorted(
        route_df["Representative_Stop"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

else:

    starts = ["Route Start"]


start = st.selectbox(
    "📍 Where are you starting?",
    starts
)


# ============================================================
# DESTINATION
# ============================================================

destinations = []


if "Destination" in route_df.columns:

    destinations = sorted(
        route_df["Destination"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


if not destinations:

    destinations = [
        "Route Destination"
    ]


destination = st.selectbox(
    "🏁 Where are you going?",
    destinations
)


# ============================================================
# TRAVEL DAY
# ============================================================

weekdays = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]


weekday = st.selectbox(
    "📅 Day of Travel",
    weekdays
)


# ============================================================
# TIME
# ============================================================

hour = st.slider(
    "⏰ Starting Time",
    min_value=0,
    max_value=23,
    value=8
)


minute = st.selectbox(
    "Minutes",
    [0, 15, 30, 45]
)


st.caption(
    f"Selected time: **{hour:02d}:{minute:02d}**"
)


# ============================================================
# CONDITIONS
# ============================================================

col1, col2 = st.columns(2)


with col1:

    weather_options = [
        "Clear",
        "Cloudy",
        "Rainy",
        "Heavy Rain"
    ]

    if "Weather" in route_df.columns:

        existing_weather = (
            route_df["Weather"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        weather_options = sorted(
            set(weather_options + existing_weather)
        )

    weather = st.selectbox(
        "🌦️ Weather",
        weather_options
    )


with col2:

    traffic_options = [
        "Low",
        "Moderate",
        "High",
        "Very High"
    ]

    if "Traffic_Level" in route_df.columns:

        existing_traffic = (
            route_df["Traffic_Level"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        traffic_options = sorted(
            set(
                traffic_options
                + existing_traffic
            )
        )

    traffic = st.selectbox(
        "🚦 Traffic Level",
        traffic_options
    )


# ============================================================
# PREDICT BUTTON
# ============================================================

predict_button = st.button(
    "🔮 Analyse My Trip",
    type="primary",
    use_container_width=True
)


# ============================================================
# RESULT
# ============================================================

if predict_button:

    result = predict_trip(
        df=df,
        route=route,
        start=start,
        destination=destination,
        weekday=weekday,
        hour=hour,
        weather=weather,
        traffic_level=traffic
    )

    delay = result["expected_delay"]

    travel_time = result["normal_travel_time"]

    total_time = result["expected_trip_time"]

    confidence = result["confidence"]

    risk = result["risk"]

    # --------------------------------------------------------
    # ETA
    # --------------------------------------------------------

    start_minutes = (
        hour * 60
        + minute
    )

    arrival_minutes = (
        start_minutes
        + total_time
    )

    arrival_minutes %= 24 * 60

    arrival_hour = int(
        arrival_minutes // 60
    )

    arrival_minute = int(
        arrival_minutes % 60
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    st.success(
        f"Historical analysis completed using "
        f"{result['sample_size']:,} similar trips."
    )

    st.markdown(
        f"""
        <div class="route-box">

        <b>📍 {start}</b>

        &nbsp;&nbsp;→&nbsp;&nbsp;

        <b>🏁 {destination}</b>

        <br><br>

        🚌 Route: <b>{route}</b>
        &nbsp;&nbsp; | &nbsp;&nbsp;
        📅 {weekday}
        &nbsp;&nbsp; | &nbsp;&nbsp;
        ⏰ {hour:02d}:{minute:02d}

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🔮 Historical Prediction")

    # --------------------------------------------------------
    # KPI ROW
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Expected Delay",
        f"{delay:.1f} min"
    )

    c2.metric(
        "Normal Travel Time",
        f"{travel_time:.0f} min"
    )

    c3.metric(
        "Expected Trip Time",
        f"{total_time:.0f} min"
    )

    c4.metric(
        "Expected Arrival",
        f"{arrival_hour:02d}:{arrival_minute:02d}"
    )

    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    st.markdown("### 🚦 Delay Risk")

    r1, r2, r3 = st.columns(3)

    r1.metric(
        "Risk Level",
        risk
    )

    r2.metric(
        "Historical Percentile",
        f"{result['delay_percentile']:.0f}%"
    )

    r3.metric(
        "On-Time Probability",
        f"{result['on_time_probability']:.0f}%"
    )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    st.progress(
        confidence / 100
    )

    st.caption(
        f"Historical confidence: **{confidence}%** "
        f"based on {result['sample_size']} similar trips."
    )

    # ========================================================
    # WHY THIS DELAY?
    # ========================================================

    st.markdown("### 🔍 Why is this delay expected?")

    factors = pd.DataFrame({
        "Historical Factor": [
            "Starting Point Average",
            "Similar Trip Average",
            "Time-of-Day Average",
            "Route Average"
        ],
        "Delay (minutes)": [
            result["start_average_delay"],
            result["similar_trip_delay"],
            result["time_period_delay"],
            result["route_average_delay"]
        ]
    })

    fig = px.bar(
        factors,
        x="Historical Factor",
        y="Delay (minutes)",
        title="Historical Delay Evidence"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ========================================================
    # BEST TIME
    # ========================================================

    best = best_time_to_travel(
        df,
        route,
        start
    )

    if best:

        st.markdown(
            f"""
            ### 💡 Smart Travel Suggestion

            Based on historical data for **{route}**
            from **{start}**, the lowest average delay
            occurs around:

            ## 🕐 {best['hour']:02d}:00

            Historical average delay:
            **{best['delay']:.1f} minutes**
            """
        )

    # ========================================================
    # WHAT IF
    # ========================================================

    st.markdown("### 🧪 What-If Travel Analysis")

    st.write(
        "What happens if you start one hour earlier "
        "or one hour later?"
    )

    what_if = what_if_analysis(
        df,
        route,
        start,
        destination,
        weekday,
        hour,
        weather,
        traffic
    )

    w1, w2, w3 = st.columns(3)

    w1.metric(
        f"One hour earlier ({max(0, hour-1):02d}:00)",
        f"{what_if['one_hour_earlier']['expected_delay']:.1f} min delay"
    )

    w2.metric(
        f"Current ({hour:02d}:00)",
        f"{what_if['current']['expected_delay']:.1f} min delay"
    )

    w3.metric(
        f"One hour later ({min(23, hour+1):02d}:00)",
        f"{what_if['one_hour_later']['expected_delay']:.1f} min delay"
    )

    # ========================================================
    # EXPLANATION
    # ========================================================

    st.markdown("### 🧠 How the prediction was calculated")

    st.info(
        "The system does not train a machine-learning model. "
        "It filters historical trips that resemble the selected "
        "route, starting point, weekday, time and conditions. "
        "It then combines historical averages using statistical "
        "weights. Delay risk is calculated using the historical "
        "percentile of delays."
    )


# ============================================================
# FDS SUMMARY
# ============================================================

st.divider()

st.header("🎓 FDS Analytical Workflow")

workflow = pd.DataFrame({
    "Stage": [
        "1. Data Collection",
        "2. Data Cleaning",
        "3. Exploratory Data Analysis",
        "4. Pattern Discovery",
        "5. Statistical Prediction",
        "6. Risk Analysis",
        "7. What-If Analysis",
        "8. Decision Support"
    ],

    "Technique": [
        "Historical transport dataset",
        "Missing values & duplicates",
        "Distribution, correlation & grouping",
        "Time, route, weather & traffic patterns",
        "Historical weighted averages",
        "Percentile-based risk",
        "Scenario comparison",
        "Smart travel recommendation"
    ]
})

st.dataframe(
    workflow,
    use_container_width=True,
    hide_index=True
)


st.caption(
    "Chennai Public Transport Analytics — "
    "FDS-based historical predictive analytics system. "
    "Weather and traffic fields are experimental/simulated where applicable."
)