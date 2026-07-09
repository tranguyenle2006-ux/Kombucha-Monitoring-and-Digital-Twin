import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
import time

st.set_page_config(page_title="Kombucha FPI Dashboard", layout="wide")
np.random.seed(7)

def minmax(s, invert=False):
    smin = s.min()
    smax = s.max()
    x = (s - smin) / (smax - smin + 1e-9)
    if invert:
        x = 1 - x
    return x.clip(0, 1)

def temp_status(t):
    if t < 25:
        return "Too cold"
    if t > 29:
        return "Too hot"
    return "Ideal"

def normalize_value(value, series, invert=False):
    smin = series.min()
    smax = series.max()
    x = (value - smin) / (smax - smin + 1e-9)
    if invert:
        x = 1 - x
    return float(np.clip(x, 0, 1))

def predict_remaining(model, features):
    features = np.asarray(features).reshape(1, -1)
    return float(np.clip(model.predict(features)[0], 0, None))

def fermentation_progress_curve(d, speed=1.0):
    x = d / (14.0 / speed)
    return np.clip(1 - np.exp(-2.2 * x), 0, 1)

def make_batch(batch_id, days=15, warm_bias=0.0, speed=1.0, noise_scale=1.0):
    d = np.arange(days)
    start_time = datetime.now() - timedelta(minutes=20 * days)
    timestamps = [start_time + timedelta(minutes=20 * i) for i in range(days)]
    prog = fermentation_progress_curve(d, speed=speed)

    pH_start = 4.15 + np.random.normal(0, 0.03)
    pH_end = 2.75 + np.random.normal(0, 0.03)
    pH = pH_start - (pH_start - pH_end) * prog + np.random.normal(0, 0.02 * noise_scale, size=days)
    pH = np.clip(pH, 2.4, 4.4)

    cond_start = 780 + np.random.normal(0, 12)
    cond_end = 1420 + np.random.normal(0, 20)
    cond = cond_start + (cond_end - cond_start) * prog + np.random.normal(0, 8 * noise_scale, size=days)

    turb_start = 55 + np.random.normal(0, 2)
    turb_end = 18 + np.random.normal(0, 1.5)
    turb = turb_start - (turb_start - turb_end) * (prog ** 0.85) + np.random.normal(0, 0.6 * noise_scale, size=days)
    turb = np.clip(turb, 0, None)

    color_start = 170 + np.random.normal(0, 2)
    color_end = 126 + np.random.normal(0, 2)
    color = color_start - (color_start - color_end) * (prog ** 0.9) + np.random.normal(0, 0.7 * noise_scale, size=days)

    room_temp = 26.2 + warm_bias + 1.4 * np.sin((d - 1) / 3.0) + np.random.normal(0, 0.25 * noise_scale, size=days)
    room_temp = np.clip(room_temp, 21, 32)

    return pd.DataFrame({
        "batch": batch_id,
        "day": d,
        "timestamp": timestamps,
        "pH": pH,
        "conductivity_uS_cm": cond,
        "turbidity_NTU": turb,
        "color_index": color,
        "room_temp_C": room_temp,
    })

@st.cache_data
def make_base_data():
    return pd.concat([
        make_batch(1, warm_bias=0.0, speed=1.0, noise_scale=1.0),
        make_batch(2, warm_bias=1.2, speed=1.1, noise_scale=1.0),
        make_batch(3, warm_bias=-1.0, speed=0.92, noise_scale=1.0),
    ], ignore_index=True)

def add_derived_fields(df):
    df = df.copy()
    df["pH_n"] = minmax(df["pH"], invert=True)
    df["cond_n"] = minmax(df["conductivity_uS_cm"], invert=False)
    df["turb_n"] = minmax(df["turbidity_NTU"], invert=True)
    df["color_n"] = minmax(df["color_index"], invert=True)

    ideal_temp = 27.0
    sigma = 3.8
    temp_eff = np.exp(-((df["room_temp_C"] - ideal_temp) ** 2) / (2 * sigma ** 2))
    temp_eff = 0.65 + 0.35 * (temp_eff - temp_eff.min()) / (temp_eff.max() - temp_eff.min() + 1e-9)
    df["temp_eff"] = temp_eff

    w_pH, w_cond, w_turb, w_color, w_temp = 0.38, 0.22, 0.18, 0.12, 0.10
    df["raw_fpi"] = (
        w_pH * df["pH_n"] +
        w_cond * df["cond_n"] +
        w_turb * df["turb_n"] +
        w_color * df["color_n"] +
        w_temp * df["temp_eff"]
    )

    max_days = 12.0
    temp_factor = np.where(
        df["room_temp_C"] < 25.0, 1.12,
        np.where(df["room_temp_C"] > 29.0, 0.88, 1.0)
    )

    df["estimated_days_remaining"] = np.clip(
        max_days * (1 - df["raw_fpi"]) * temp_factor,
        0,
        max_days
    )
    df["fpi_progress_pct"] = np.clip(100 * (1 - df["estimated_days_remaining"] / max_days), 0, 100)
    df["temp_status"] = df["room_temp_C"].apply(temp_status)
    return df

def recalibrate_fpi(df):
    df = df.copy()
    df["actual_days_remaining"] = (df.groupby("batch")["day"].transform("max") - df["day"]).astype(float)

    batch_stats = []
    for b in sorted(df["batch"].unique()):
        bdf = df[df["batch"] == b]
        mae = mean_absolute_error(bdf["actual_days_remaining"], bdf["estimated_days_remaining"])
        bias = (bdf["actual_days_remaining"] - bdf["estimated_days_remaining"]).mean()
        batch_stats.append({"batch": b, "mae": mae, "bias": bias})

    stats_df = pd.DataFrame(batch_stats)
    avg_bias = stats_df["bias"].iloc[:-1].mean() if len(stats_df) > 1 else 0.0
    df["estimated_days_remaining"] = np.clip(df["estimated_days_remaining"] + 0.55 * avg_bias, 0, 12.0)
    df["fpi_progress_pct"] = np.clip(100 * (1 - df["estimated_days_remaining"] / 12.0), 0, 100)
    return df, stats_df

def batch_learning_curve(df):
    results = []
    batch_ids = sorted(df["batch"].unique())
    for i in range(1, len(batch_ids) + 1):
        subset = df[df["batch"].isin(batch_ids[:i])].copy()
        X_local = subset[["pH_n", "cond_n", "turb_n", "color_n", "room_temp_C", "temp_eff"]]
        y_local = (subset.groupby("batch")["day"].transform("max") - subset["day"]).astype(float)
        model_local = LinearRegression().fit(X_local, y_local)
        preds = np.clip(model_local.predict(X_local), 0, None)
        mae = mean_absolute_error(y_local, preds)
        rmse = np.sqrt(mean_squared_error(y_local, preds))
        results.append({"batch_count": i, "reg_mae": mae, "reg_rmse": rmse})
    return pd.DataFrame(results)

def get_live_data(base_df, batch_id, model=None):
    view = base_df[base_df["batch"] == batch_id].copy().reset_index(drop=True)
    session_key = f"ghost_sensors_{batch_id}"
    if session_key not in st.session_state:
        st.session_state[session_key] = {"data_points": []}

    ghost_state = st.session_state[session_key]
    point_index = len(ghost_state["data_points"])
    max_idx = len(view) - 1
    progress = min(point_index / (max_idx * 4), 1.0)

    if progress < 1.0 and len(ghost_state["data_points"]) < max_idx * 5:
        base_idx = min(int(progress * max_idx), max_idx - 1)
        next_idx = min(base_idx + 1, max_idx)
        blend = (progress * max_idx) - base_idx

        new_reading = {
            "batch": batch_id,
            "timestamp": datetime.now() + timedelta(minutes=20 * point_index),
            "pH": view.iloc[base_idx]["pH"] + (view.iloc[next_idx]["pH"] - view.iloc[base_idx]["pH"]) * blend + np.random.normal(0, 0.01),
            "conductivity_uS_cm": view.iloc[base_idx]["conductivity_uS_cm"] + (view.iloc[next_idx]["conductivity_uS_cm"] - view.iloc[base_idx]["conductivity_uS_cm"]) * blend + np.random.normal(0, 4),
            "turbidity_NTU": view.iloc[base_idx]["turbidity_NTU"] + (view.iloc[next_idx]["turbidity_NTU"] - view.iloc[base_idx]["turbidity_NTU"]) * blend + np.random.normal(0, 0.4),
            "color_index": view.iloc[base_idx]["color_index"] + (view.iloc[next_idx]["color_index"] - view.iloc[base_idx]["color_index"]) * blend + np.random.normal(0, 0.4),
            "room_temp_C": view.iloc[base_idx]["room_temp_C"] + (view.iloc[next_idx]["room_temp_C"] - view.iloc[base_idx]["room_temp_C"]) * blend + np.random.normal(0, 0.1),
        }

        new_reading["pH_n"] = normalize_value(new_reading["pH"], view["pH"], invert=True)
        new_reading["cond_n"] = normalize_value(new_reading["conductivity_uS_cm"], view["conductivity_uS_cm"], invert=False)
        new_reading["turb_n"] = normalize_value(new_reading["turbidity_NTU"], view["turbidity_NTU"], invert=True)
        new_reading["color_n"] = normalize_value(new_reading["color_index"], view["color_index"], invert=True)

        ideal_temp = 27.0
        sigma = 3.8
        temp_eff = np.exp(-((new_reading["room_temp_C"] - ideal_temp) ** 2) / (2 * sigma ** 2))
        new_reading["temp_eff"] = 0.65 + 0.35 * temp_eff

        w_pH, w_cond, w_turb, w_color, w_temp = 0.38, 0.22, 0.18, 0.12, 0.10
        new_reading["raw_fpi"] = (
            w_pH * new_reading["pH_n"] +
            w_cond * new_reading["cond_n"] +
            w_turb * new_reading["turb_n"] +
            w_color * new_reading["color_n"] +
            w_temp * new_reading["temp_eff"]
        )

        max_days = 12.0
        temp_factor = 1.12 if new_reading["room_temp_C"] < 25.0 else (0.88 if new_reading["room_temp_C"] > 29.0 else 1.0)
        new_reading["estimated_days_remaining"] = np.clip(
            max_days * (1 - new_reading["raw_fpi"]) * temp_factor,
            0,
            max_days
        )
        new_reading["fpi_progress_pct"] = np.clip(100 * (1 - new_reading["estimated_days_remaining"] / max_days), 0, 100)
        new_reading["temp_status"] = temp_status(new_reading["room_temp_C"])
        new_reading["day"] = float(view.iloc[base_idx]["day"] + blend)

        if model is not None:
            new_reading["pred_remaining_reg"] = predict_remaining(
                model,
                [
                    new_reading["pH_n"],
                    new_reading["cond_n"],
                    new_reading["turb_n"],
                    new_reading["color_n"],
                    new_reading["room_temp_C"],
                    new_reading["temp_eff"],
                ],
            )

        ghost_state["data_points"].append(new_reading)

    if ghost_state["data_points"]:
        live_df = pd.DataFrame(ghost_state["data_points"])
        return pd.concat([view, live_df], ignore_index=True)

    return view

if "extra_batch_params" not in st.session_state:
    st.session_state["extra_batch_params"] = []
if "batch_start_id" not in st.session_state:
    st.session_state["batch_start_id"] = 4

base_df = make_base_data()

with st.sidebar:
    if st.button("Add new batch"):
        st.session_state["extra_batch_params"].append({
            "warm_bias": float(np.random.uniform(-1.5, 1.5)),
            "speed": float(np.random.uniform(0.88, 1.12)),
            "noise_scale": float(np.random.uniform(0.85, 1.25)),
        })
        st.rerun()

    if st.session_state["extra_batch_params"]:
        if st.button("Remove last batch"):
            st.session_state["extra_batch_params"].pop()
            st.rerun()
        if st.button("Clear extra batches"):
            st.session_state["extra_batch_params"].clear()
            st.rerun()

extra_batches = []
for idx, params in enumerate(st.session_state["extra_batch_params"]):
    batch_id = st.session_state["batch_start_id"] + idx
    extra_batches.append(
        make_batch(
            batch_id,
            warm_bias=params["warm_bias"],
            speed=params["speed"],
            noise_scale=params["noise_scale"]
        )
    )

if extra_batches:
    df = pd.concat([base_df] + extra_batches, ignore_index=True)
else:
    df = base_df.copy()

df = add_derived_fields(df)
df, fpi_stats = recalibrate_fpi(df)

targets = (df.groupby("batch")["day"].transform("max") - df["day"]).astype(float)
X = df[["pH_n", "cond_n", "turb_n", "color_n", "room_temp_C", "temp_eff"]]
y = targets
model = LinearRegression().fit(X, y)
df["pred_remaining_reg"] = np.clip(model.predict(X), 0, None)

learning_df = batch_learning_curve(df)

st.title("Kombucha Fermentation Control Panel")
st.caption("Synthetic kombucha monitoring with FPI, regression, and batch-wise learning")

with st.sidebar:
    auto_refresh = st.checkbox("Live Update", value=True)
    refresh_rate = st.selectbox("Update Interval", [1, 2, 3, 5], index=1)
    batch = st.selectbox("Batch", sorted(df["batch"].unique()))
    st.markdown("---")
    st.markdown("### Model Notes")
    st.write("FPI is calibrated from historical batch error.")
    st.write("Regression retrains after each new batch.")
    st.write(f"Extra batches: {len(st.session_state['extra_batch_params'])}")

view = get_live_data(df, batch, model)
view = view.copy()

if view["pred_remaining_reg"].isna().any():
    missing_idx = view["pred_remaining_reg"].isna()
    view.loc[missing_idx, "pred_remaining_reg"] = np.clip(
        model.predict(view.loc[missing_idx, X.columns]), 0, None
    )

view["actual_days_remaining"] = (view.groupby("batch")["day"].transform("max") - view["day"]).astype(float)
view["fpi_abs_error"] = (view["actual_days_remaining"] - view["estimated_days_remaining"]).abs()
view["reg_abs_error"] = (view["actual_days_remaining"] - view["pred_remaining_reg"]).abs()
view["fpi_pct_error"] = 100 * view["fpi_abs_error"] / (view["actual_days_remaining"] + 1e-9)
view["reg_pct_error"] = 100 * view["reg_abs_error"] / (view["actual_days_remaining"] + 1e-9)

latest = view.iloc[-1]

fpi_mae = mean_absolute_error(view["actual_days_remaining"], view["estimated_days_remaining"])
fpi_rmse = np.sqrt(mean_squared_error(view["actual_days_remaining"], view["estimated_days_remaining"]))
reg_mae = mean_absolute_error(view["actual_days_remaining"], view["pred_remaining_reg"])
reg_rmse = np.sqrt(mean_squared_error(view["actual_days_remaining"], view["pred_remaining_reg"]))

c1, c2, c3, c4 = st.columns(4)
c1.metric("pH", f'{latest["pH"]:.2f}')
c2.metric("Conductivity", f'{latest["conductivity_uS_cm"]:.0f} µS/cm')
c3.metric("Turbidity", f'{latest["turbidity_NTU"]:.1f} NTU')
c4.metric("Room Temp", f'{latest["room_temp_C"]:.1f} °C', latest["temp_status"])

c5, c6, c7, c8 = st.columns(4)
c5.metric("Color Index", f'{latest["color_index"]:.1f}')
c6.metric("FPI Progress", f'{latest["fpi_progress_pct"]:.1f}%')
c7.metric("Estimated ETA", f'{latest["estimated_days_remaining"]:.1f} days')
c8.metric("Regression ETA", f'{latest["pred_remaining_reg"]:.1f} days')

st.subheader("Progress Comparison")
p1, p2 = st.columns(2)
with p1:
    st.metric("Hand-built FPI", f'{latest["fpi_progress_pct"]:.1f}%')
    st.progress(float(latest["fpi_progress_pct"]) / 100.0, text=f'FPI Progress: {latest["fpi_progress_pct"]:.1f}%')
with p2:
    reg_progress_pct = np.clip(100 * (1 - float(latest["pred_remaining_reg"]) / 12.0), 0, 100)
    st.metric("Regression", f'{reg_progress_pct:.1f}%')
    st.progress(reg_progress_pct / 100.0, text=f"Regression Progress: {reg_progress_pct:.1f}%")

st.subheader("Validation Summary")
m1, m2, m3, m4 = st.columns(4)
m1.metric("FPI MAE", f"{fpi_mae:.2f} days")
m2.metric("FPI RMSE", f"{fpi_rmse:.2f} days")
m3.metric("Regression MAE", f"{reg_mae:.2f} days")
m4.metric("Regression RMSE", f"{reg_rmse:.2f} days")

summary = pd.DataFrame({
    "method": ["FPI", "Regression"],
    "mean_abs_error_days": [fpi_mae, reg_mae],
    "rmse_days": [fpi_rmse, reg_rmse],
    "mean_pct_error": [view["fpi_pct_error"].mean(), view["reg_pct_error"].mean()]
})
st.dataframe(summary, use_container_width=True)

st.subheader("FPI Calibration by Batch")
st.dataframe(fpi_stats, use_container_width=True)

st.subheader("Batch Learning Curve")
fig_curve = go.Figure()
fig_curve.add_trace(go.Scatter(x=learning_df["batch_count"], y=learning_df["reg_mae"], mode="lines+markers", name="Regression MAE"))
fig_curve.add_trace(go.Scatter(x=learning_df["batch_count"], y=learning_df["reg_rmse"], mode="lines+markers", name="Regression RMSE"))
fig_curve.update_layout(title="Regression Error vs Number of Batches", xaxis_title="Batch Count", yaxis_title="Error (days)", height=400)
st.plotly_chart(fig_curve, use_container_width=True)

st.subheader("Error Trends")
e1, e2 = st.columns(2)
with e1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view["timestamp"], y=view["fpi_abs_error"], mode="lines+markers", name="FPI Abs Error"))
    fig.update_layout(title="FPI Absolute Error", xaxis_title="Time", yaxis_title="Days", height=350)
    st.plotly_chart(fig, use_container_width=True)
with e2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view["timestamp"], y=view["reg_abs_error"], mode="lines+markers", name="Regression Abs Error"))
    fig.update_layout(title="Regression Absolute Error", xaxis_title="Time", yaxis_title="Days", height=350)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Live Sensor Charts")
g1, g2 = st.columns(2)
with g1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view["timestamp"], y=view["pH"], mode="lines+markers", name="pH"))
    fig.update_layout(title="pH Level", xaxis_title="Time", yaxis_title="pH", height=400)
    st.plotly_chart(fig, use_container_width=True)
with g2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view["timestamp"], y=view["conductivity_uS_cm"], mode="lines+markers", name="Conductivity"))
    fig.update_layout(title="Conductivity", xaxis_title="Time", yaxis_title="µS/cm", height=400)
    st.plotly_chart(fig, use_container_width=True)

g3, g4 = st.columns(2)
with g3:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view["timestamp"], y=view["turbidity_NTU"], mode="lines+markers", name="Turbidity"))
    fig.update_layout(title="Turbidity", xaxis_title="Time", yaxis_title="NTU", height=400)
    st.plotly_chart(fig, use_container_width=True)
with g4:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view["timestamp"], y=view["color_index"], mode="lines+markers", name="Color Index"))
    fig.update_layout(title="Color Index", xaxis_title="Time", yaxis_title="Index", height=400)
    st.plotly_chart(fig, use_container_width=True)

g5, g6 = st.columns(2)
with g5:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view["timestamp"], y=view["room_temp_C"], mode="lines+markers", name="Room Temp"))
    fig.update_layout(title="Room Temperature", xaxis_title="Time", yaxis_title="°C", height=400)
    st.plotly_chart(fig, use_container_width=True)
with g6:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view["timestamp"], y=view["estimated_days_remaining"], mode="lines+markers", name="FPI ETA"))
    fig.update_layout(title="Estimated Days Remaining", xaxis_title="Time", yaxis_title="Days", height=400)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Regression Coefficients")
coef_df = pd.DataFrame({"feature": X.columns, "coefficient": model.coef_}).sort_values("coefficient", ascending=False)
st.dataframe(coef_df, use_container_width=True)

st.subheader("Full Dataset")
st.dataframe(view, use_container_width=True)

if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()