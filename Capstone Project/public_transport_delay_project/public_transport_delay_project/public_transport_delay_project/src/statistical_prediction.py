"""
statistical_prediction.py

FDS-based statistical prediction for Chennai Public Transport.

NO MACHINE LEARNING IS USED.

The system predicts expected delay using historical observations
with similar travel conditions.

Methods:
    - Conditional historical mean
    - Median
    - Standard deviation
    - Percentiles
    - IQR
    - 95% confidence interval
    - Historical similarity analysis
    - Delay risk classification
    - What-if analysis
"""

import pandas as pd
import numpy as np


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_mean(series):
    if series is None or len(series) == 0:
        return 0.0
    return float(series.mean())


def safe_median(series):
    if series is None or len(series) == 0:
        return 0.0
    return float(series.median())


def safe_std(series):
    if series is None or len(series) < 2:
        return 0.0
    return float(series.std())


# ============================================================
# HISTORICAL SIMILARITY
# ============================================================

def find_similar_records(
    df,
    hour=None,
    weather=None,
    traffic_level=None,
    weekday=None,
    time_period=None,
    tolerance=1
):
    """
    Find historical trips having conditions similar to
    the selected scenario.

    The search starts strict and can gradually relax the
    hour condition using tolerance.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    result = df.copy()

    # --------------------------------------------------------
    # Hour
    # --------------------------------------------------------

    if hour is not None and "Hour" in result.columns:

        result = result[
            result["Hour"].between(
                float(hour) - tolerance,
                float(hour) + tolerance
            )
        ]

    # --------------------------------------------------------
    # Weather
    # --------------------------------------------------------

    if (
        weather is not None
        and weather != "Any"
        and "Weather" in result.columns
    ):
        result = result[
            result["Weather"].astype(str) == str(weather)
        ]

    # --------------------------------------------------------
    # Traffic
    # --------------------------------------------------------

    if (
        traffic_level is not None
        and traffic_level != "Any"
        and "Traffic_Level" in result.columns
    ):
        result = result[
            result["Traffic_Level"].astype(str)
            == str(traffic_level)
        ]

    # --------------------------------------------------------
    # Weekday
    # --------------------------------------------------------

    if (
        weekday is not None
        and weekday != "Any"
        and "Weekday" in result.columns
    ):
        result = result[
            result["Weekday"].astype(str)
            == str(weekday)
        ]

    # --------------------------------------------------------
    # Time period
    # --------------------------------------------------------

    if (
        time_period is not None
        and time_period != "Any"
        and "Time_Period" in result.columns
    ):
        result = result[
            result["Time_Period"].astype(str)
            == str(time_period)
        ]

    return result


# ============================================================
# STATISTICAL PREDICTION
# ============================================================

def statistical_prediction(
    df,
    hour=None,
    weather=None,
    traffic_level=None,
    weekday=None,
    time_period=None
):
    """
    Estimate delay using historical data.

    NO ML MODEL IS USED.

    Returns:
        dictionary containing statistical results.
    """

    if df is None or df.empty:
        return {
            "success": False,
            "message": "Dataset is empty."
        }

    if "Delay_Minutes" not in df.columns:
        return {
            "success": False,
            "message": "Delay_Minutes column not found."
        }

    # --------------------------------------------------------
    # First try very similar historical trips
    # --------------------------------------------------------

    similar = find_similar_records(
        df,
        hour=hour,
        weather=weather,
        traffic_level=traffic_level,
        weekday=weekday,
        time_period=time_period,
        tolerance=1
    )

    # --------------------------------------------------------
    # If too few records, relax hour
    # --------------------------------------------------------

    if len(similar) < 20:

        similar = find_similar_records(
            df,
            hour=hour,
            weather=weather,
            traffic_level=traffic_level,
            weekday=weekday,
            time_period=time_period,
            tolerance=2
        )

    # --------------------------------------------------------
    # If still too few, use wider historical group
    # --------------------------------------------------------

    if len(similar) < 20:

        similar = find_similar_records(
            df,
            hour=hour,
            weather=weather,
            traffic_level=traffic_level,
            weekday=weekday,
            time_period=None,
            tolerance=2
        )

    # --------------------------------------------------------
    # Final fallback
    # --------------------------------------------------------

    if len(similar) < 10:
        similar = df.copy()

    delay = pd.to_numeric(
        similar["Delay_Minutes"],
        errors="coerce"
    ).dropna()

    if len(delay) == 0:
        return {
            "success": False,
            "message": "No valid delay observations found."
        }

    # --------------------------------------------------------
    # DESCRIPTIVE STATISTICS
    # --------------------------------------------------------

    mean_delay = float(delay.mean())
    median_delay = float(delay.median())
    std_delay = float(delay.std()) if len(delay) > 1 else 0.0

    minimum = float(delay.min())
    maximum = float(delay.max())

    q1 = float(delay.quantile(0.25))
    q3 = float(delay.quantile(0.75))

    p90 = float(delay.quantile(0.90))
    p95 = float(delay.quantile(0.95))

    # --------------------------------------------------------
    # 95% CONFIDENCE INTERVAL
    # --------------------------------------------------------

    n = len(delay)

    if n > 1:
        standard_error = std_delay / np.sqrt(n)

        margin = 1.96 * standard_error

        ci_lower = mean_delay - margin
        ci_upper = mean_delay + margin

    else:
        ci_lower = mean_delay
        ci_upper = mean_delay

    ci_lower = max(0, ci_lower)

    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    overall_delay = float(
        pd.to_numeric(
            df["Delay_Minutes"],
            errors="coerce"
        ).median()
    )

    if mean_delay <= overall_delay * 0.90:

        risk = "LOW"

    elif mean_delay <= overall_delay * 1.20:

        risk = "MEDIUM"

    else:

        risk = "HIGH"

    # --------------------------------------------------------
    # RISK SCORE
    # --------------------------------------------------------

    if overall_delay > 0:

        risk_score = (
            mean_delay / overall_delay
        ) * 50

        risk_score = min(
            100,
            max(0, risk_score)
        )

    else:

        risk_score = 0

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {
        "success": True,

        "records_used": int(n),

        "mean_delay": round(mean_delay, 2),

        "median_delay": round(median_delay, 2),

        "std_delay": round(std_delay, 2),

        "minimum_delay": round(minimum, 2),

        "maximum_delay": round(maximum, 2),

        "q1": round(q1, 2),

        "q3": round(q3, 2),

        "p90": round(p90, 2),

        "p95": round(p95, 2),

        "ci_lower": round(ci_lower, 2),

        "ci_upper": round(ci_upper, 2),

        "risk": risk,

        "risk_score": round(risk_score, 1),

        "historical_data": similar
    }


# ============================================================
# DELAY RISK
# ============================================================

def calculate_delay_risk(delay, historical_delays):
    """
    Calculate risk based on percentile position.
    """

    if historical_delays is None or len(historical_delays) == 0:
        return "UNKNOWN"

    historical_delays = pd.Series(
        historical_delays
    ).dropna()

    percentile = (
        historical_delays <= delay
    ).mean() * 100

    if percentile >= 80:
        return "HIGH"

    elif percentile >= 50:
        return "MEDIUM"

    return "LOW"


# ============================================================
# WHAT-IF ANALYSIS
# ============================================================

def what_if_analysis(
    df,
    base_conditions,
    changed_conditions
):
    """
    Compare two historical scenarios.

    Example:

    Base:
        8 AM + High Traffic + Rainy

    What-if:
        10 AM + Medium Traffic + Rainy
    """

    base = statistical_prediction(
        df,
        hour=base_conditions.get("hour"),
        weather=base_conditions.get("weather"),
        traffic_level=base_conditions.get("traffic_level"),
        weekday=base_conditions.get("weekday"),
        time_period=base_conditions.get("time_period")
    )

    scenario = statistical_prediction(
        df,
        hour=changed_conditions.get("hour"),
        weather=changed_conditions.get("weather"),
        traffic_level=changed_conditions.get("traffic_level"),
        weekday=changed_conditions.get("weekday"),
        time_period=changed_conditions.get("time_period")
    )

    if not base["success"] or not scenario["success"]:
        return {
            "success": False
        }

    difference = (
        base["mean_delay"]
        - scenario["mean_delay"]
    )

    percentage_change = 0

    if base["mean_delay"] != 0:

        percentage_change = (
            difference
            / base["mean_delay"]
        ) * 100

    if difference > 0:

        interpretation = (
            f"Expected delay may decrease by "
            f"{abs(difference):.1f} minutes."
        )

    elif difference < 0:

        interpretation = (
            f"Expected delay may increase by "
            f"{abs(difference):.1f} minutes."
        )

    else:

        interpretation = (
            "Both scenarios have similar historical delay."
        )

    return {
        "success": True,

        "base_delay": base["mean_delay"],

        "scenario_delay": scenario["mean_delay"],

        "difference": round(difference, 2),

        "percentage_change": round(
            percentage_change,
            2
        ),

        "interpretation": interpretation,

        "base": base,

        "scenario": scenario
    }


# ============================================================
# ROUTE RELIABILITY
# ============================================================

def route_reliability(df):

    if "Delay_Minutes" not in df.columns:
        return pd.DataFrame()

    possible_routes = [
        "Route",
        "Route_ID",
        "Route_Name",
        "Bus_Route"
    ]

    route_column = None

    for col in possible_routes:

        if col in df.columns:
            route_column = col
            break

    if route_column is None:
        return pd.DataFrame()

    result = (
        df.groupby(route_column)["Delay_Minutes"]
        .agg(
            Trips="count",
            Average_Delay="mean",
            Median_Delay="median",
            Std_Deviation="std"
        )
        .reset_index()
    )

    result["Reliability_Score"] = (
        100 /
        (1 + result["Std_Deviation"].fillna(0))
    )

    result["Reliability_Score"] = (
        result["Reliability_Score"]
        .clip(0, 100)
        .round(1)
    )

    return result.sort_values(
        "Reliability_Score",
        ascending=False
    )


# ============================================================
# IQR ANOMALY DETECTION
# ============================================================

def detect_delay_anomalies(df):

    if "Delay_Minutes" not in df.columns:
        return pd.DataFrame()

    result = df.copy()

    delay = pd.to_numeric(
        result["Delay_Minutes"],
        errors="coerce"
    )

    q1 = delay.quantile(0.25)
    q3 = delay.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    result["Anomaly"] = (
        (delay < lower)
        |
        (delay > upper)
    )

    result["Anomaly_Type"] = np.where(
        delay > upper,
        "Extreme High Delay",
        np.where(
            delay < lower,
            "Unusually Low Delay",
            "Normal"
        )
    )

    return result


# ============================================================
# HOURLY STATISTICS
# ============================================================

def hourly_statistics(df):

    if (
        "Hour" not in df.columns
        or "Delay_Minutes" not in df.columns
    ):
        return pd.DataFrame()

    result = (
        df.groupby("Hour")["Delay_Minutes"]
        .agg(
            Average="mean",
            Median="median",
            Std_Deviation="std",
            Trips="count"
        )
        .reset_index()
    )

    return result


# ============================================================
# TRAFFIC STATISTICS
# ============================================================

def traffic_statistics(df):

    if (
        "Traffic_Level" not in df.columns
        or "Delay_Minutes" not in df.columns
    ):
        return pd.DataFrame()

    return (
        df.groupby("Traffic_Level")["Delay_Minutes"]
        .agg(
            Average="mean",
            Median="median",
            Std_Deviation="std",
            Trips="count"
        )
        .reset_index()
    )


# ============================================================
# WEATHER STATISTICS
# ============================================================

def weather_statistics(df):

    if (
        "Weather" not in df.columns
        or "Delay_Minutes" not in df.columns
    ):
        return pd.DataFrame()

    return (
        df.groupby("Weather")["Delay_Minutes"]
        .agg(
            Average="mean",
            Median="median",
            Std_Deviation="std",
            Trips="count"
        )
        .reset_index()
    )