import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =========================================================
# REGRESSION METRICS
# =========================================================

def evaluate_regression_model(actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(actual, predicted)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# =========================================================
# ACTUAL VS PREDICTED
# =========================================================

def actual_vs_predicted(actual, predicted):

    result = pd.DataFrame({
        "Actual": actual,
        "Predicted": predicted
    })

    fig = px.scatter(
        result,
        x="Actual",
        y="Predicted",
        title="Actual vs Predicted Delay"
    )

    minimum = min(
        result["Actual"].min(),
        result["Predicted"].min()
    )

    maximum = max(
        result["Actual"].max(),
        result["Predicted"].max()
    )

    fig.add_trace(
        go.Scatter(
            x=[minimum, maximum],
            y=[minimum, maximum],
            mode="lines",
            name="Perfect Prediction"
        )
    )

    fig.update_layout(
        xaxis_title="Actual Delay (Minutes)",
        yaxis_title="Predicted Delay (Minutes)"
    )

    return fig


# =========================================================
# RESIDUAL ANALYSIS
# =========================================================

def residual_plot(actual, predicted):

    residuals = (
        np.array(actual) -
        np.array(predicted)
    )

    result = pd.DataFrame({
        "Predicted": predicted,
        "Residual": residuals
    })

    fig = px.scatter(
        result,
        x="Predicted",
        y="Residual",
        title="Residual Analysis"
    )

    fig.add_hline(y=0)

    fig.update_layout(
        xaxis_title="Predicted Delay (Minutes)",
        yaxis_title="Residual"
    )

    return fig


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

def feature_importance(model, feature_names):

    if not hasattr(model, "feature_importances_"):
        return None

    importance = model.feature_importances_

    if len(importance) != len(feature_names):
        return None

    result = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    return result.sort_values(
        "Importance",
        ascending=False
    )


# =========================================================
# FEATURE IMPORTANCE GRAPH
# =========================================================

def feature_importance_plot(model, feature_names):

    result = feature_importance(
        model,
        feature_names
    )

    if result is None:
        return None

    result = result.head(15)

    fig = px.bar(
        result,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Feature Importance"
    )

    fig.update_layout(
        yaxis={
            "categoryorder": "total ascending"
        }
    )

    return fig


# =========================================================
# MODEL COMPARISON FILE
# =========================================================

def load_model_comparison(path):

    try:

        df = pd.read_csv(path)

        return df

    except Exception:

        return pd.DataFrame()