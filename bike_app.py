import math
from pathlib import Path
from typing import List, Tuple
import altair as alt


import joblib
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Bike Demand & Revenue Optimization",
    page_icon="🚲",
    layout="wide",
)

st.title("🚲 Bike Demand & Revenue Optimization")
st.markdown(
    "Predict daily bike rental demand, estimate an optimal price, and compare baseline vs optimized revenue."
)


# -----------------------------
# Paths and constants
# -----------------------------
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "model.joblib"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.joblib"

BASE_PRICE = 5.0
DEFAULT_ALPHA = 0.15
PRICE_MIN = 3.0
PRICE_MAX = 8.0
PRICE_STEP = 0.25

SEASON_MAP = {
    "Spring": 1,
    "Summer": 2,
    "Fall": 3,
    "Winter": 4,
}

WEATHER_MAP = {
    "Clear / Few Clouds": 1,
    "Mist / Cloudy": 2,
    "Light Snow or Rain": 3,
}

WEEKDAY_MAP = {
    "Sunday": 0,
    "Monday": 1,
    "Tuesday": 2,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6,
}


# -----------------------------
# Helper functions
# -----------------------------
@st.cache_resource

def load_artifacts() -> Tuple[object, List[str]]:
    if not MODEL_PATH.exists() or not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Make sure models/model.joblib and models/feature_columns.joblib exist."
        )

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    return model, feature_columns


def build_features(
    yr: int,
    holiday: int,
    workingday: int,
    temp: float,
    atemp: float,
    hum: float,
    windspeed: float,
    month: int,
    season: int,
    weekday: int,
    weathersit: int,
    feature_columns: List[str],
) -> pd.DataFrame:
    data = {
        "yr": yr,
        "holiday": holiday,
        "workingday": workingday,
        "temp": temp,
        "atemp": atemp,
        "hum": hum,
        "windspeed": windspeed,
        "month_sin": math.sin(2 * math.pi * month / 12),
        "month_cos": math.cos(2 * math.pi * month / 12),
        f"season_{season}": 1,
        f"weekday_{weekday}": 1,
        f"weathersit_{weathersit}": 1,
    }

    row = pd.DataFrame([data])
    row = row.reindex(columns=feature_columns, fill_value=0)
    return row


def adjusted_demand(
    predicted_demand: float,
    price: float,
    base_price: float = BASE_PRICE,
    alpha: float = DEFAULT_ALPHA,
) -> float:
    demand = predicted_demand * (1 - alpha * (price - base_price))
    return max(float(demand), 0.0)


def revenue(
    predicted_demand: float,
    price: float,
    base_price: float = BASE_PRICE,
    alpha: float = DEFAULT_ALPHA,
) -> float:
    return float(price * adjusted_demand(predicted_demand, price, base_price, alpha))


def find_best_price(
    predicted_demand: float,
    price_min: float = PRICE_MIN,
    price_max: float = PRICE_MAX,
    step: float = PRICE_STEP,
    base_price: float = BASE_PRICE,
    alpha: float = DEFAULT_ALPHA,
) -> Tuple[float, float, float, pd.DataFrame]:
    prices = np.arange(price_min, price_max + step, step)

    records = []
    best_price = price_min
    best_revenue = -np.inf
    best_adjusted_demand = 0.0

    for price in prices:
        adj_demand = adjusted_demand(predicted_demand, price, base_price, alpha)
        rev = revenue(predicted_demand, price, base_price, alpha)
        records.append(
            {
                "price": float(price),
                "adjusted_demand": float(adj_demand),
                "revenue": float(rev),
            }
        )

        if rev > best_revenue:
            best_price = float(price)
            best_revenue = float(rev)
            best_adjusted_demand = float(adj_demand)

    curve_df = pd.DataFrame(records)
    return best_price, best_adjusted_demand, best_revenue, curve_df


# -----------------------------
# Load model
# -----------------------------
try:
    model, feature_columns = load_artifacts()
except Exception as exc:
    st.error(f"Could not load model artifacts: {exc}")
    st.stop()


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Inputs")

season_label = st.sidebar.selectbox("Season", list(SEASON_MAP.keys()), index=1)
month = st.sidebar.slider("Month", min_value=1, max_value=12, value=7)
weekday_label = st.sidebar.selectbox("Weekday", list(WEEKDAY_MAP.keys()), index=1)
weathersit_label = st.sidebar.selectbox("Weather Situation", list(WEATHER_MAP.keys()), index=0)

yr = st.sidebar.selectbox("Year", options=[0, 1], index=1, format_func=lambda x: "2011" if x == 0 else "2012")
holiday = st.sidebar.selectbox("Holiday", options=[0, 1], index=0)
workingday = st.sidebar.selectbox("Working Day", options=[0, 1], index=1)

temp = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.60, step=0.01)
atemp = st.sidebar.slider("Feels Like Temperature", min_value=0.0, max_value=1.0, value=0.58, step=0.01)
hum = st.sidebar.slider("Humidity", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
windspeed = st.sidebar.slider("Windspeed", min_value=0.0, max_value=1.0, value=0.20, step=0.01)

st.sidebar.header("Pricing Assumptions")
base_price = st.sidebar.slider("Base Price ($)", min_value=1.0, max_value=10.0, value=BASE_PRICE, step=0.25)
alpha = st.sidebar.slider("Elasticity (alpha)", min_value=0.01, max_value=0.50, value=DEFAULT_ALPHA, step=0.01)
price_min = st.sidebar.slider("Minimum Candidate Price ($)", min_value=1.0, max_value=10.0, value=PRICE_MIN, step=0.25)
price_max = st.sidebar.slider("Maximum Candidate Price ($)", min_value=1.0, max_value=12.0, value=PRICE_MAX, step=0.25)
price_step = st.sidebar.select_slider("Price Step ($)", options=[0.1, 0.25, 0.5, 1.0], value=PRICE_STEP)

if price_max <= price_min:
    st.sidebar.error("Maximum candidate price must be greater than minimum candidate price.")
    st.stop()


# -----------------------------
# Prediction
# -----------------------------
features = build_features(
    yr=yr,
    holiday=holiday,
    workingday=workingday,
    temp=temp,
    atemp=atemp,
    hum=hum,
    windspeed=windspeed,
    month=month,
    season=SEASON_MAP[season_label],
    weekday=WEEKDAY_MAP[weekday_label],
    weathersit=WEATHER_MAP[weathersit_label],
    feature_columns=feature_columns,
)

predicted_baseline_demand = float(model.predict(features)[0])
optimal_price, expected_adjusted_demand, expected_revenue, curve_df = find_best_price(
    predicted_demand=predicted_baseline_demand,
    price_min=price_min,
    price_max=price_max,
    step=price_step,
    base_price=base_price,
    alpha=alpha,
)

baseline_revenue = float(base_price * predicted_baseline_demand)
revenue_gain = expected_revenue - baseline_revenue
revenue_gain_pct = (revenue_gain / baseline_revenue * 100) if baseline_revenue > 0 else 0.0


# -----------------------------
# Main layout
# -----------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Predicted Baseline Demand", f"{predicted_baseline_demand:,.0f}")
col2.metric("Optimal Price", f"${optimal_price:,.2f}")
col3.metric("Expected Revenue", f"${expected_revenue:,.2f}", delta=f"{revenue_gain_pct:.2f}%")
col4.metric("Expected Adjusted Demand", f"{expected_adjusted_demand:,.0f}")

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Revenue Curve")

    chart = alt.Chart(curve_df).mark_line(point=True).encode(
        x=alt.X("price:Q", title="Price ($)"),
        y=alt.Y(
            "revenue:Q",
            title="Revenue ($)",
            axis=alt.Axis(format="$,.2f")
        ),
        tooltip=[
            alt.Tooltip("price:Q", format="$.2f"),
            alt.Tooltip("revenue:Q", format="$,.2f")
        ]
    ).properties(
        height=250,
        width=420
    )

    st.altair_chart(chart, use_container_width=False)

    st.subheader("Price Simulation Table")

    formatted_df = curve_df.copy()
    formatted_df["price"] = formatted_df["price"].map("${:,.2f}".format)
    formatted_df["revenue"] = formatted_df["revenue"].map("${:,.2f}".format)
    formatted_df["adjusted_demand"] = formatted_df["adjusted_demand"].map("{:,.2f}".format)

    st.dataframe(
        formatted_df,
        use_container_width=False,
        height=250
    )

with right:
    st.subheader("Scenario Summary")
    st.write(
        {
            "predicted_baseline_demand": f"{predicted_baseline_demand:,.2f}",
            "base_price": f"${base_price:,.2f}",
            "baseline_revenue": f"${baseline_revenue:,.2f}",
            "optimal_price": f"${optimal_price:,.2f}",
            "expected_adjusted_demand": f"{expected_adjusted_demand:,.2f}",
            "expected_revenue": f"${expected_revenue:,.2f}",
            "estimated_revenue_gain": f"${revenue_gain:,.2f}",
            "estimated_revenue_gain_pct": f"{revenue_gain_pct:,.2f}%",
            "alpha": round(alpha, 2),
        }
    )

    st.subheader("Input Notes")
    st.caption("Holiday: 0 = No, 1 = Yes")
    st.caption("Working Day: 0 = No, 1 = Yes")
    
st.markdown("---")
st.caption(
    "Note: pricing results are based on a simulated elasticity assumption rather than observed historical pricing data."
)
