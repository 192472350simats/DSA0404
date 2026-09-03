"""
historical_predictor.py

Statistical Predictive Analytics for Chennai Public Transport.

IMPORTANT:
This module DOES NOT use Machine Learning.

Prediction is based on:
1. Historical route averages
2. Historical stop averages
3. Time-of-day patterns
4. Weekday patterns
5. Traffic conditions
6. Weather conditions
7. Percentile-based risk
8. Weighted historical similarity

This is a Statistical / FDS Predictive Analytics approach.
"""

import pandas as pd
import numpy as np


# ============================================================
# BASIC HELPERS
# ============================================================

def _clean_numeric(series, default=0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _safe_mean(df, column, default=0):
    if df.empty or column not in df.columns:
        return float(default)

    values = pd.to_numeric(df[column], errors="coerce").dropna()

    if values.empty:
        return float(default)

    return float(values.mean())


def _safe_median(df, column, default=0):
    if df.empty or column not in df.columns:
        return float(default)

    values = pd.to_numeric(df[column], errors="coerce").dropna()

    if values.empty:
        return float(default)

    return float(values.median())


# ============================================================
# TIME PERIOD
# ============================================================

def get_time_period(hour):

    if 5 <= hour < 9:
        return "Morning Peak"

    if 9 <= hour < 12:
        return "Mid-Morning"

    if 12 <= hour < 17:
        return "Afternoon Peak"

    if 17 <= hour < 20:
        return "Evening"

    if 20 <= hour < 24:
        return "Late Night"

    return "Night"


# ============================================================
# HISTORICAL SIMILARITY FILTER
# ============================================================

def find_similar_trips(
    df,
    route=None,
    start=None,
    weekday=None,
    hour=None,
    weather=None,
    traffic_level=None
):
    """
    Find historical trips similar to the user's selected journey.

    This is NOT machine learning.
    It is rule-based statistical filtering.
    """

    data = df.copy()

    if "Route_Name" in data.columns and route:
        route_data = data[data["Route_Name"] == route]

        if not route_data.empty:
            data = route_data

    # --------------------------------------------------------
    # Starting point
    # --------------------------------------------------------

    if start and "Representative_Stop" in data.columns:

        start_data = data[
            data["Representative_Stop"].astype(str) == str(start)
        ]

        if len(start_data) >= 5:
            data = start_data

    # --------------------------------------------------------
    # Weekday
    # --------------------------------------------------------

    if weekday and "Weekday" in data.columns:

        weekday_data = data[
            data["Weekday"].astype(str) == str(weekday)
        ]

        if len(weekday_data) >= 5:
            data = weekday_data

    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    if hour is not None and "Hour" in data.columns:

        hour_values = pd.to_numeric(
            data["Hour"],
            errors="coerce"
        )

        time_data = data[
            hour_values.between(hour - 1, hour + 1)
        ]

        if len(time_data) >= 5:
            data = time_data

    # --------------------------------------------------------
    # Weather
    # --------------------------------------------------------

    if weather and "Weather" in data.columns:

        weather_data = data[
            data["Weather"].astype(str) == str(weather)
        ]

        if len(weather_data) >= 5:
            data = weather_data

    # --------------------------------------------------------
    # Traffic
    # --------------------------------------------------------

    if traffic_level and "Traffic_Level" in data.columns:

        traffic_data = data[
            data["Traffic_Level"].astype(str)
            == str(traffic_level)
        ]

        if len(traffic_data) >= 5:
            data = traffic_data

    return data


# ============================================================
# HISTORICAL DELAY ESTIMATION
# ============================================================

def estimate_delay(
    df,
    route,
    start,
    weekday,
    hour,
    weather,
    traffic_level
):
    """
    Calculate expected delay using weighted historical averages.

    NO ML.
    """

    if "Delay_Minutes" not in df.columns:
        return {
            "delay": 0,
            "base_delay": 0,
            "similar_delay": 0,
            "route_delay": 0,
            "confidence": 0,
            "sample_size": 0
        }

    route_data = df[
        df["Route_Name"] == route
    ].copy()

    # --------------------------------------------------------
    # Route average
    # --------------------------------------------------------

    route_delay = _safe_mean(
        route_data,
        "Delay_Minutes",
        0
    )

    # --------------------------------------------------------
    # Start-point average
    # --------------------------------------------------------

    start_data = route_data[
        route_data["Representative_Stop"].astype(str)
        == str(start)
    ]

    start_delay = _safe_mean(
        start_data,
        "Delay_Minutes",
        route_delay
    )

    # --------------------------------------------------------
    # Similar trips
    # --------------------------------------------------------

    similar = find_similar_trips(
        df,
        route=route,
        start=start,
        weekday=weekday,
        hour=hour,
        weather=weather,
        traffic_level=traffic_level
    )

    similar_delay = _safe_mean(
        similar,
        "Delay_Minutes",
        start_delay
    )

    # --------------------------------------------------------
    # Time-of-day delay
    # --------------------------------------------------------

    time_data = route_data[
        pd.to_numeric(
            route_data["Hour"],
            errors="coerce"
        ).between(hour - 1, hour + 1)
    ]

    time_delay = _safe_mean(
        time_data,
        "Delay_Minutes",
        route_delay
    )

    # --------------------------------------------------------
    # Statistical weighted estimate
    # --------------------------------------------------------

    # More weight is given to trips that are
    # more similar to the user's situation.

    if len(similar) >= 20:

        estimated_delay = (
            0.55 * similar_delay +
            0.25 * start_delay +
            0.12 * time_delay +
            0.08 * route_delay
        )

    elif len(similar) >= 5:

        estimated_delay = (
            0.40 * similar_delay +
            0.30 * start_delay +
            0.20 * time_delay +
            0.10 * route_delay
        )

    else:

        estimated_delay = (
            0.50 * start_delay +
            0.30 * time_delay +
            0.20 * route_delay
        )

    estimated_delay = max(
        0,
        round(float(estimated_delay), 1)
    )

    # --------------------------------------------------------
    # Confidence based on historical sample size
    # --------------------------------------------------------

    n = len(similar)

    if n >= 100:
        confidence = 95

    elif n >= 50:
        confidence = 90

    elif n >= 20:
        confidence = 82

    elif n >= 10:
        confidence = 72

    elif n >= 5:
        confidence = 60

    else:
        confidence = 45

    return {
        "delay": estimated_delay,
        "base_delay": round(start_delay, 1),
        "similar_delay": round(similar_delay, 1),
        "time_delay": round(time_delay, 1),
        "route_delay": round(route_delay, 1),
        "confidence": confidence,
        "sample_size": n
    }


# ============================================================
# DELAY RISK
# ============================================================

def calculate_delay_risk(df, route, start, delay):

    route_data = df[
        df["Route_Name"] == route
    ]

    start_data = route_data[
        route_data["Representative_Stop"].astype(str)
        == str(start)
    ]

    values = pd.to_numeric(
        start_data["Delay_Minutes"],
        errors="coerce"
    ).dropna()

    if len(values) < 5:

        values = pd.to_numeric(
            route_data["Delay_Minutes"],
            errors="coerce"
        ).dropna()

    if values.empty:

        return {
            "risk": "Unknown",
            "percentile": 50
        }

    percentile = (
        (values <= delay).sum()
        / len(values)
    ) * 100

    if percentile >= 85:
        risk = "Very High"

    elif percentile >= 65:
        risk = "High"

    elif percentile >= 40:
        risk = "Moderate"

    else:
        risk = "Low"

    return {
        "risk": risk,
        "percentile": round(percentile, 1)
    }


# ============================================================
# ON-TIME PROBABILITY
# ============================================================

def calculate_on_time_probability(
    df,
    route,
    start
):

    data = df[
        (df["Route_Name"] == route) &
        (
            df["Representative_Stop"].astype(str)
            == str(start)
        )
    ]

    if data.empty:

        data = df[
            df["Route_Name"] == route
        ]

    if data.empty:

        return 0

    delays = pd.to_numeric(
        data["Delay_Minutes"],
        errors="coerce"
    )

    valid = delays.dropna()

    if valid.empty:

        return 0

    # On-time = delay <= 5 minutes

    probability = (
        (valid <= 5).sum()
        / len(valid)
    ) * 100

    return round(float(probability), 1)


# ============================================================
# TRAVEL TIME
# ============================================================

def estimate_travel_time(
    df,
    route,
    start,
    destination
):

    route_data = df[
        df["Route_Name"] == route
    ]

    # --------------------------------------------------------
    # If a travel-time column exists
    # --------------------------------------------------------

    possible_columns = [
        "Travel_Time_Minutes",
        "Travel_Time",
        "Journey_Time",
        "Duration_Minutes"
    ]

    for column in possible_columns:

        if column in route_data.columns:

            value = _safe_median(
                route_data,
                column,
                45
            )

            return round(value, 1)

    # --------------------------------------------------------
    # Estimate from distance if available
    # --------------------------------------------------------

    if "Distance_km_Proxy" in route_data.columns:

        distance = _safe_median(
            route_data,
            "Distance_km_Proxy",
            15
        )

        # Approximate urban bus speed
        speed = 20

        travel_time = (
            distance / speed
        ) * 60

        return round(
            max(10, travel_time),
            1
        )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    return 45.0


# ============================================================
# COMPLETE PREDICTION
# ============================================================

def predict_trip(
    df,
    route,
    start,
    destination,
    weekday,
    hour,
    weather="Clear",
    traffic_level="Moderate"
):

    delay_info = estimate_delay(
        df=df,
        route=route,
        start=start,
        weekday=weekday,
        hour=hour,
        weather=weather,
        traffic_level=traffic_level
    )

    travel_time = estimate_travel_time(
        df,
        route,
        start,
        destination
    )

    risk_info = calculate_delay_risk(
        df,
        route,
        start,
        delay_info["delay"]
    )

    on_time_probability = (
        calculate_on_time_probability(
            df,
            route,
            start
        )
    )

    expected_trip_time = (
        travel_time +
        delay_info["delay"]
    )

    return {
        "expected_delay": delay_info["delay"],
        "normal_travel_time": travel_time,
        "expected_trip_time": round(
            expected_trip_time,
            1
        ),
        "risk": risk_info["risk"],
        "delay_percentile": risk_info["percentile"],
        "confidence": delay_info["confidence"],
        "sample_size": delay_info["sample_size"],
        "on_time_probability": on_time_probability,
        "route_average_delay": delay_info["route_delay"],
        "start_average_delay": delay_info["base_delay"],
        "similar_trip_delay": delay_info["similar_delay"],
        "time_period_delay": delay_info["time_delay"]
    }


# ============================================================
# BEST TIME TO TRAVEL
# ============================================================

def best_time_to_travel(
    df,
    route,
    start
):

    data = df[
        (df["Route_Name"] == route) &
        (
            df["Representative_Stop"].astype(str)
            == str(start)
        )
    ].copy()

    if data.empty:
        return None

    if "Hour" not in data.columns:
        return None

    result = (
        data.groupby("Hour")["Delay_Minutes"]
        .mean()
        .reset_index()
        .sort_values("Delay_Minutes")
    )

    if result.empty:
        return None

    best = result.iloc[0]

    return {
        "hour": int(best["Hour"]),
        "delay": round(
            float(best["Delay_Minutes"]),
            1
        )
    }


# ============================================================
# WHAT-IF ANALYSIS
# ============================================================

def what_if_analysis(
    df,
    route,
    start,
    destination,
    weekday,
    hour,
    weather,
    traffic_level
):

    current = predict_trip(
        df,
        route,
        start,
        destination,
        weekday,
        hour,
        weather,
        traffic_level
    )

    # Compare with one hour earlier
    earlier_hour = max(0, hour - 1)

    earlier = predict_trip(
        df,
        route,
        start,
        destination,
        weekday,
        earlier_hour,
        weather,
        traffic_level
    )

    # Compare with one hour later
    later_hour = min(23, hour + 1)

    later = predict_trip(
        df,
        route,
        start,
        destination,
        weekday,
        later_hour,
        weather,
        traffic_level
    )

    return {
        "current": current,
        "one_hour_earlier": earlier,
        "one_hour_later": later
    }