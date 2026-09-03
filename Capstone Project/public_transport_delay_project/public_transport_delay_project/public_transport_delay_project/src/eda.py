import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# DATASET OVERVIEW
# =========================================================

def dataset_overview(df):

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum())
    }


# =========================================================
# DATA TYPES
# =========================================================

def datatype_summary(df):

    return pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values,
        "Unique Values": df.nunique().values
    })


# =========================================================
# DESCRIPTIVE STATISTICS
# =========================================================

def descriptive_statistics(df):

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.empty:
        return pd.DataFrame()

    return numeric_df.describe().T


# =========================================================
# MISSING VALUE ANALYSIS
# =========================================================

def missing_value_summary(df):

    missing = df.isnull().sum()

    result = pd.DataFrame({
        "Column": missing.index,
        "Missing Values": missing.values
    })

    result["Missing Percentage"] = (
        result["Missing Values"] / len(df) * 100
    ).round(2)

    return result[result["Missing Values"] > 0]


# =========================================================
# DELAY DISTRIBUTION
# =========================================================

def fig_delay_distribution(df):

    if "Delay_Minutes" not in df.columns:
        return None

    fig = px.histogram(
        df,
        x="Delay_Minutes",
        nbins=30,
        title="Distribution of Transport Delay",
        labels={"Delay_Minutes": "Delay (Minutes)"}
    )

    fig.update_layout(
        xaxis_title="Delay (Minutes)",
        yaxis_title="Number of Trips"
    )

    return fig


# =========================================================
# DELAY BOXPLOT
# =========================================================

def fig_delay_boxplot(df):

    if "Delay_Minutes" not in df.columns:
        return None

    fig = px.box(
        df,
        y="Delay_Minutes",
        title="Delay Outlier Analysis"
    )

    fig.update_layout(
        yaxis_title="Delay (Minutes)"
    )

    return fig


# =========================================================
# TRAFFIC VS DELAY
# =========================================================

def fig_traffic_vs_delay(df):

    if "Traffic_Index" not in df.columns:
        return None

    if "Delay_Minutes" not in df.columns:
        return None

    fig = px.scatter(
        df,
        x="Traffic_Index",
        y="Delay_Minutes",
        trendline="ols",
        title="Traffic Index vs Transport Delay",
        labels={
            "Traffic_Index": "Traffic Index",
            "Delay_Minutes": "Delay (Minutes)"
        }
    )

    return fig


# =========================================================
# RAINFALL VS DELAY
# =========================================================

def fig_rainfall_vs_delay(df):

    if "Rainfall_mm" not in df.columns:
        return None

    if "Delay_Minutes" not in df.columns:
        return None

    fig = px.scatter(
        df,
        x="Rainfall_mm",
        y="Delay_Minutes",
        trendline="ols",
        title="Rainfall vs Transport Delay",
        labels={
            "Rainfall_mm": "Rainfall (mm)",
            "Delay_Minutes": "Delay (Minutes)"
        }
    )

    return fig


# =========================================================
# WEATHER VS DELAY
# =========================================================

def fig_weather_vs_delay(df):

    if "Weather" not in df.columns:
        return None

    if "Delay_Minutes" not in df.columns:
        return None

    fig = px.box(
        df,
        x="Weather",
        y="Delay_Minutes",
        title="Delay Distribution by Weather",
        labels={
            "Weather": "Weather Condition",
            "Delay_Minutes": "Delay (Minutes)"
        }
    )

    return fig


# =========================================================
# TRAFFIC LEVEL VS DELAY
# =========================================================

def fig_traffic_level_vs_delay(df):

    if "Traffic_Level" not in df.columns:
        return None

    if "Delay_Minutes" not in df.columns:
        return None

    fig = px.box(
        df,
        x="Traffic_Level",
        y="Delay_Minutes",
        title="Delay Distribution by Traffic Level",
        labels={
            "Traffic_Level": "Traffic Level",
            "Delay_Minutes": "Delay (Minutes)"
        }
    )

    return fig


# =========================================================
# HOURLY DELAY
# =========================================================

def fig_hourly_delay_fds(df):

    if "Hour" not in df.columns:
        return None

    if "Delay_Minutes" not in df.columns:
        return None

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

    fig.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Average Delay (Minutes)"
    )

    return fig


# =========================================================
# CORRELATION MATRIX
# =========================================================

def correlation_matrix(df):

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.empty:
        return pd.DataFrame()

    return numeric_df.corr()


# =========================================================
# CORRELATION HEATMAP
# =========================================================

def fig_correlation_heatmap(df):

    corr = correlation_matrix(df)

    if corr.empty:
        return None

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            colorscale="RdBu",
            zmin=-1,
            zmax=1
        )
    )

    fig.update_layout(
        title="Correlation Matrix of Numerical Features"
    )

    return fig


# =========================================================
# OUTLIER SUMMARY
# =========================================================

def outlier_summary(df):

    numeric_df = df.select_dtypes(include=np.number)

    results = []

    for column in numeric_df.columns:

        q1 = numeric_df[column].quantile(0.25)
        q3 = numeric_df[column].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = (
            (numeric_df[column] < lower) |
            (numeric_df[column] > upper)
        ).sum()

        results.append({
            "Feature": column,
            "Outliers": int(count),
            "Outlier Percentage":
                round(count / len(df) * 100, 2)
        })

    return pd.DataFrame(results)


# =========================================================
# NUMERICAL FEATURES
# =========================================================

def numerical_features(df):

    return df.select_dtypes(include=np.number).columns.tolist()


# =========================================================
# CATEGORICAL FEATURES
# =========================================================

def categorical_features(df):

    return df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()


# =========================================================
# 🚨 NEW FEATURE 1
# DELAY ANOMALY DETECTION
# =========================================================

def delay_anomaly_analysis(df):

    if "Delay_Minutes" not in df.columns:
        return pd.DataFrame()

    data = df.copy()

    mean_delay = data["Delay_Minutes"].mean()
    std_delay = data["Delay_Minutes"].std()

    if std_delay == 0 or pd.isna(std_delay):
        data["Anomaly_Score"] = 0
        data["Anomaly"] = False
        return data

    data["Anomaly_Score"] = (
        (data["Delay_Minutes"] - mean_delay) /
        std_delay
    ).abs()

    data["Anomaly"] = data["Anomaly_Score"] >= 2

    return data


# =========================================================
# ANOMALY SUMMARY
# =========================================================

def anomaly_summary(df):

    data = delay_anomaly_analysis(df)

    if data.empty:
        return {
            "anomalies": 0,
            "percentage": 0
        }

    anomalies = int(data["Anomaly"].sum())

    percentage = round(
        anomalies / len(data) * 100,
        2
    )

    return {
        "anomalies": anomalies,
        "percentage": percentage
    }


# =========================================================
# ANOMALY VISUALIZATION
# =========================================================

def fig_delay_anomalies(df):

    data = delay_anomaly_analysis(df)

    if data.empty:
        return None

    data = data.reset_index(drop=True)

    data["Trip"] = data.index + 1

    fig = px.scatter(
        data,
        x="Trip",
        y="Delay_Minutes",
        color="Anomaly",
        title="🚨 Transport Delay Anomaly Detection",
        labels={
            "Trip": "Trip Number",
            "Delay_Minutes": "Delay (Minutes)",
            "Anomaly": "Anomalous Delay"
        }
    )

    return fig


# =========================================================
# 🎯 NEW FEATURE 2
# DELAY RISK SCORE
# =========================================================

def delay_risk_score(
    predicted_delay,
    historical_average=None,
    traffic_index=None,
    rainfall=None
):

    score = 0

    # -----------------------------------------------------
    # Prediction-based component
    # -----------------------------------------------------

    if predicted_delay >= 30:
        score += 45

    elif predicted_delay >= 20:
        score += 35

    elif predicted_delay >= 10:
        score += 20

    elif predicted_delay > 5:
        score += 10

    # -----------------------------------------------------
    # Historical comparison
    # -----------------------------------------------------

    if (
        historical_average is not None
        and historical_average > 0
    ):

        ratio = predicted_delay / historical_average

        if ratio >= 2:
            score += 25

        elif ratio >= 1.5:
            score += 18

        elif ratio >= 1.2:
            score += 10

    # -----------------------------------------------------
    # Traffic component
    # -----------------------------------------------------

    if traffic_index is not None:

        if traffic_index >= 80:
            score += 20

        elif traffic_index >= 60:
            score += 12

        elif traffic_index >= 40:
            score += 6

    # -----------------------------------------------------
    # Rainfall component
    # -----------------------------------------------------

    if rainfall is not None:

        if rainfall >= 20:
            score += 10

        elif rainfall >= 10:
            score += 6

        elif rainfall > 0:
            score += 3

    score = min(100, int(score))

    if score <= 30:
        category = "LOW"

    elif score <= 60:
        category = "MODERATE"

    elif score <= 80:
        category = "HIGH"

    else:
        category = "VERY HIGH"

    return score, category


# =========================================================
# 🚌 NEW FEATURE 3
# ROUTE RELIABILITY INDEX
# =========================================================

def route_reliability_index(df):

    required = ["Delay_Minutes"]

    if not all(column in df.columns for column in required):
        return pd.DataFrame()

    data = df.copy()

    # -----------------------------------------------------
    # Route column detection
    # -----------------------------------------------------

    route_column = None

    possible_columns = [
        "Route",
        "Route_ID",
        "Route_Name",
        "route"
    ]

    for column in possible_columns:

        if column in data.columns:
            route_column = column
            break

    if route_column is None:
        return pd.DataFrame()

    result = (
        data.groupby(route_column)
        .agg(
            Average_Delay=("Delay_Minutes", "mean"),
            Delay_Std=("Delay_Minutes", "std"),
            Trips=("Delay_Minutes", "count")
        )
        .reset_index()
    )

    # -----------------------------------------------------
    # On-time percentage
    # -----------------------------------------------------

    on_time = (
        data.assign(
            On_Time=data["Delay_Minutes"] <= 5
        )
        .groupby(route_column)["On_Time"]
        .mean()
        .reset_index()
    )

    on_time["On_Time_Percentage"] = (
        on_time["On_Time"] * 100
    ).round(2)

    on_time = on_time.drop(columns=["On_Time"])

    result = result.merge(
        on_time,
        on=route_column,
        how="left"
    )

    # -----------------------------------------------------
    # Reliability score
    # -----------------------------------------------------

    result["Reliability_Score"] = (
        100
        - result["Average_Delay"].clip(0, 30) / 30 * 50
        - result["Delay_Std"].fillna(0).clip(0, 20) / 20 * 25
        + result["On_Time_Percentage"] / 100 * 25
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


# =========================================================
# ROUTE RELIABILITY CHART
# =========================================================

def fig_route_reliability(df):

    reliability = route_reliability_index(df)

    if reliability.empty:
        return None

    route_column = reliability.columns[0]

    fig = px.bar(
        reliability,
        x=route_column,
        y="Reliability_Score",
        title="🚌 Route Reliability Index",
        labels={
            route_column: "Route",
            "Reliability_Score": "Reliability Score"
        }
    )

    fig.update_yaxes(range=[0, 100])

    return fig


# =========================================================
# 🕐 NEW FEATURE 4
# BEST AND WORST TRAVEL TIME
# =========================================================

def best_worst_travel_time(df):

    if "Hour" not in df.columns:
        return None

    if "Delay_Minutes" not in df.columns:
        return None

    hourly = (
        df.groupby("Hour")["Delay_Minutes"]
        .mean()
        .reset_index()
    )

    if hourly.empty:
        return None

    best = hourly.loc[
        hourly["Delay_Minutes"].idxmin()
    ]

    worst = hourly.loc[
        hourly["Delay_Minutes"].idxmax()
    ]

    return {
        "best_hour": int(best["Hour"]),
        "best_delay": round(
            float(best["Delay_Minutes"]),
            2
        ),
        "worst_hour": int(worst["Hour"]),
        "worst_delay": round(
            float(worst["Delay_Minutes"]),
            2
        )
    }


# =========================================================
# 🕐 TRAVEL TIME CHART
# =========================================================

def fig_travel_time_recommendation(df):

    if "Hour" not in df.columns:
        return None

    if "Delay_Minutes" not in df.columns:
        return None

    hourly = (
        df.groupby("Hour")["Delay_Minutes"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        hourly,
        x="Hour",
        y="Delay_Minutes",
        title="🕐 Average Delay by Travel Time",
        labels={
            "Hour": "Hour of Day",
            "Delay_Minutes": "Average Delay (Minutes)"
        }
    )

    return fig


# =========================================================
# 🔍 NEW FEATURE 5
# DELAY FACTOR ANALYSIS
# =========================================================

def delay_factor_analysis(df):

    if "Delay_Minutes" not in df.columns:
        return pd.DataFrame()

    results = []

    # -----------------------------------------------------
    # Traffic
    # -----------------------------------------------------

    if "Traffic_Index" in df.columns:

        correlation = df[
            ["Traffic_Index", "Delay_Minutes"]
        ].corr().iloc[0, 1]

        results.append({
            "Factor": "Traffic",
            "Correlation": round(
                float(correlation),
                3
            ),
            "Impact": abs(float(correlation))
        })

    # -----------------------------------------------------
    # Rainfall
    # -----------------------------------------------------

    if "Rainfall_mm" in df.columns:

        correlation = df[
            ["Rainfall_mm", "Delay_Minutes"]
        ].corr().iloc[0, 1]

        results.append({
            "Factor": "Rainfall",
            "Correlation": round(
                float(correlation),
                3
            ),
            "Impact": abs(float(correlation))
        })

    # -----------------------------------------------------
    # Hour
    # -----------------------------------------------------

    if "Hour" in df.columns:

        correlation = df[
            ["Hour", "Delay_Minutes"]
        ].corr().iloc[0, 1]

        results.append({
            "Factor": "Time of Day",
            "Correlation": round(
                float(correlation),
                3
            ),
            "Impact": abs(float(correlation))
        })

    result = pd.DataFrame(results)

    if not result.empty:

        result = result.sort_values(
            "Impact",
            ascending=False
        )

    return result


# =========================================================
# FACTOR IMPORTANCE CHART
# =========================================================

def fig_delay_factor_analysis(df):

    result = delay_factor_analysis(df)

    if result.empty:
        return None

    fig = px.bar(
        result,
        x="Factor",
        y="Impact",
        title="🔍 Factors Associated With Transport Delay",
        labels={
            "Impact": "Absolute Correlation"
        }
    )

    return fig


# =========================================================
# 🧪 NEW FEATURE 6
# WHAT-IF DATAFRAME
# =========================================================

def what_if_comparison(
    current_delay,
    traffic_delay=None,
    no_rain_delay=None,
    ideal_delay=None
):

    scenarios = [
        ("Current Conditions", current_delay)
    ]

    if traffic_delay is not None:
        scenarios.append(
            ("Moderate Traffic", traffic_delay)
        )

    if no_rain_delay is not None:
        scenarios.append(
            ("No Rainfall", no_rain_delay)
        )

    if ideal_delay is not None:
        scenarios.append(
            ("Ideal Conditions", ideal_delay)
        )

    result = pd.DataFrame(
        scenarios,
        columns=[
            "Scenario",
            "Predicted Delay"
        ]
    )

    result["Predicted Delay"] = (
        result["Predicted Delay"]
        .round(2)
    )

    result["Improvement"] = (
        current_delay -
        result["Predicted Delay"]
    ).round(2)

    return result


# =========================================================
# WHAT-IF CHART
# =========================================================

def fig_what_if_comparison(
    current_delay,
    traffic_delay=None,
    no_rain_delay=None,
    ideal_delay=None
):

    result = what_if_comparison(
        current_delay,
        traffic_delay,
        no_rain_delay,
        ideal_delay
    )

    fig = px.bar(
        result,
        x="Scenario",
        y="Predicted Delay",
        text="Predicted Delay",
        title="🧪 What-If Delay Simulation",
        labels={
            "Predicted Delay":
                "Predicted Delay (Minutes)"
        }
    )

    fig.update_traces(
        textposition="outside"
    )

    return fig


# =========================================================
# 📊 HISTORICAL DELAY FOR ROUTE
# =========================================================

def historical_route_delay(
    df,
    route
):

    route_column = None

    for column in [
        "Route",
        "Route_ID",
        "Route_Name",
        "route"
    ]:

        if column in df.columns:
            route_column = column
            break

    if route_column is None:
        return None

    filtered = df[
        df[route_column].astype(str)
        == str(route)
    ]

    if filtered.empty:
        return None

    return float(
        filtered["Delay_Minutes"].mean()
    )


# =========================================================
# 📈 DELAY CATEGORY
# =========================================================

def delay_category(delay):

    if delay <= 5:
        return "🟢 On Time"

    elif delay <= 10:
        return "🟡 Slight Delay"

    elif delay <= 20:
        return "🟠 Moderate Delay"

    else:
        return "🔴 Severe Delay"


# =========================================================
# 💡 AUTOMATIC DATA INSIGHTS
# =========================================================

def automatic_insights(df):

    insights = []

    if "Delay_Minutes" not in df.columns:
        return insights

    average_delay = df[
        "Delay_Minutes"
    ].mean()

    max_delay = df[
        "Delay_Minutes"
    ].max()

    insights.append(
        f"Average transport delay is "
        f"{average_delay:.2f} minutes."
    )

    insights.append(
        f"Maximum recorded delay is "
        f"{max_delay:.2f} minutes."
    )

    # -----------------------------------------------------
    # Traffic insight
    # -----------------------------------------------------

    if (
        "Traffic_Index" in df.columns
        and len(df) > 1
    ):

        corr = df[
            ["Traffic_Index", "Delay_Minutes"]
        ].corr().iloc[0, 1]

        if corr > 0.3:

            insights.append(
                "Traffic shows a positive association "
                "with transport delay."
            )

    # -----------------------------------------------------
    # Rainfall insight
    # -----------------------------------------------------

    if (
        "Rainfall_mm" in df.columns
        and len(df) > 1
    ):

        corr = df[
            ["Rainfall_mm", "Delay_Minutes"]
        ].corr().iloc[0, 1]

        if corr > 0.3:

            insights.append(
                "Rainfall shows a positive association "
                "with transport delay."
            )

    # -----------------------------------------------------
    # Peak-hour insight
    # -----------------------------------------------------

    if "Hour" in df.columns:

        hourly = (
            df.groupby("Hour")[
                "Delay_Minutes"
            ]
            .mean()
        )

        if not hourly.empty:

            worst_hour = hourly.idxmax()

            insights.append(
                f"Hour {int(worst_hour)} has the "
                f"highest average delay."
            )

    return insights