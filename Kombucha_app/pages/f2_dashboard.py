import sys
from pathlib import Path

# Allow importing arduino_reader from parent folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import arduino_reader
import time


# ----- Temperature efficiency & days remaining (F2) -----
def temp_efficiency(T, T_ideal=27, sigma=3.8):
    if T is None or (isinstance(T, float) and math.isnan(T)):
        T = T_ideal
    eff = math.exp(-(T - T_ideal)**2 / (2 * sigma**2))
    return 0.65 + 0.35 * eff


def days_remaining_f2(fpi, D_max, T_current, T_ideal=27):
    if T_current is None or (isinstance(T_current, float) and math.isnan(T_current)):
        T_current = T_ideal
    temp_adjust = 1 + 0.02 * (T_ideal - T_current)
    return max(0, D_max * (1 - fpi) * temp_adjust)


# ----- Synthetic batch generator -----
def generate_f2_batch(batch_id, days=8, ideal_temp=27, temp_std=1.0,
                      pressure_end=1.8, water_start=100, water_end=95):
    np.random.seed(batch_id)
    t = np.linspace(0, days, 40)
    temp = ideal_temp + np.random.normal(0, temp_std, len(t))
    temp = np.clip(temp, 20, 35)
    pressure = pressure_end * (1 - np.exp(-0.8 * t)) + np.random.normal(0, 0.05, len(t))
    pressure = np.clip(pressure, 0, 2.5)
    water = water_start - (water_start - water_end) * (t / days) + np.random.normal(0, 0.5, len(t))
    water = np.clip(water, 80, 100)
    df = pd.DataFrame({
        'batch': batch_id,
        'day': t,
        'temperature': temp,
        'pressure': pressure,
        'water_level': water
    })
    return df


# ----- Live simulation (fallback) -----
def get_live_f2():
    if 'f2_sim_state' not in st.session_state:
        st.session_state.f2_sim_state = {'day': 0, 'batch': 1}
    state = st.session_state.f2_sim_state
    t = state['day']
    pressure = min(2.5, 1.8 * (1 - np.exp(-0.8 * t)) + np.random.normal(0, 0.05))
    temp = 27 + np.random.normal(0, 1.5)
    water = max(80, 100 - 0.6 * t + np.random.normal(0, 0.5))
    state['day'] += 0.1
    return {'temperature': temp, 'pressure': pressure, 'water_level': water}


# ----- FPI calculation for F2 (water + temp, pressure optional) -----
def compute_f2_fpi(pressure, water, temp, batch_data=None):
    w_min, w_max = 80, 100
    p_min, p_max = 0, 2.0

    if batch_data is not None and len(batch_data) >= 2:
        if not pd.isna(water):
            w_vals = batch_data['water_level'].dropna()
            if len(w_vals) > 0:
                w_min, w_max = w_vals.min(), w_vals.max()
        if pressure is not None and not pd.isna(pressure):
            p_vals = batch_data['pressure'].dropna()
            if len(p_vals) > 0:
                p_min, p_max = p_vals.min(), p_vals.max()

    if not pd.isna(water) and w_max != w_min:
        w_n = 1 - (water - w_min) / (w_max - w_min)
    else:
        w_n = 0.5

    w_n = max(0, min(1, w_n))

    if pressure is not None and not pd.isna(pressure) and p_max != p_min:
        p_n = (pressure - p_min) / (p_max - p_min)
        p_n = max(0, min(1, p_n))
    else:
        p_n = 0.5

    w_p, w_w = 0.30, 0.40
    raw_fpi = w_p * p_n + w_w * w_n
    temp_fac = temp_efficiency(temp)
    fpi = raw_fpi * temp_fac
    return fpi, p_n, w_n


def f2_alerts(pressure, temp, water, fpi, days_left, day):
    alerts = []
    if pressure is not None and not np.isnan(pressure) and pressure > 2.0:
        alerts.append("💥 Pressure > 2 bar – over‑carbonation risk! Refrigerate or burp.")
    if pressure is not None and not np.isnan(pressure) and pressure < 0.3 and day > 2:
        alerts.append("🫧 Very low pressure – possible low yeast activity or leak.")
    if temp is not None and not np.isnan(temp) and temp < 25:
        alerts.append("❄️ Temperature low – carbonation will be slow.")
    if temp is not None and not np.isnan(temp) and temp > 29:
        alerts.append("🔥 Temperature high – risk of off‑flavours and excessive pressure.")
    if water is not None and not np.isnan(water) and water < 80:
        alerts.append("💧 Water level very low – check for leaks.")
    return alerts


def validate_f2_model(synthetic_data):
    D_max = 8
    df = synthetic_data.copy()
    df['actual_days_remaining'] = D_max - df['day']
    
    fpi_preds = []
    for _, row in df.iterrows():
        batch_data = df[df['batch'] == row['batch']]
        fpi, _, _ = compute_f2_fpi(row['pressure'], row['water_level'], row['temperature'], batch_data)
        days = days_remaining_f2(fpi, D_max, row['temperature'])
        fpi_preds.append(days)
    
    norm_p = (df['pressure'] - df['pressure'].min()) / (df['pressure'].max() - df['pressure'].min())
    norm_w = 1 - (df['water_level'] - df['water_level'].min()) / (df['water_level'].max() - df['water_level'].min())
    temp_eff = df['temperature'].apply(temp_efficiency)
    X = pd.DataFrame({'pressure_n': norm_p, 'water_n': norm_w, 'temp_eff': temp_eff})
    y = df['actual_days_remaining']
    model = LinearRegression().fit(X, y)
    ml_preds = model.predict(X)
    
    rmse_fpi = math.sqrt(mean_squared_error(y, fpi_preds))
    mae_fpi = mean_absolute_error(y, fpi_preds)
    r2_fpi = r2_score(y, fpi_preds)
    
    rmse_ml = math.sqrt(mean_squared_error(y, ml_preds))
    mae_ml = mean_absolute_error(y, ml_preds)
    r2_ml = r2_score(y, ml_preds)
    
    return {'FPI model': {'RMSE': rmse_fpi, 'MAE': mae_fpi, 'R²': r2_fpi},
            'ML model': {'RMSE': rmse_ml, 'MAE': mae_ml, 'R²': r2_ml}}, model


# ----- Streamlit page -----
def show():
    st.title("🍾 Second Fermentation (F2) Monitor")
    
    if 'f2_batch_data' not in st.session_state:
        st.session_state.f2_batch_data = pd.DataFrame(
            columns=['day','temperature','pressure','water_level']
        )
    if 'f2_batch_id' not in st.session_state:
        st.session_state.f2_batch_id = 1
    if 'arduino_serial' not in st.session_state:
        st.session_state.arduino_serial = None

    # Stable live mode: last valid reading + failure count
    if 'last_valid_arduino_reading_f2' not in st.session_state:
        st.session_state.last_valid_arduino_reading_f2 = None
    if 'consecutive_failures_f2' not in st.session_state:
        st.session_state.consecutive_failures_f2 = 0

    st.sidebar.header("F2 Controls")
    if st.sidebar.button("Start New F2 Batch"):
        st.session_state.f2_batch_id += 1
        st.session_state.f2_batch_data = pd.DataFrame(
            columns=['day','temperature','pressure','water_level']
        )
        st.session_state.f2_sim_state = {'day':0, 'batch': st.session_state.f2_batch_id}

    sim_mode = st.sidebar.checkbox("Simulate live data", value=True)

    # --- Get current sensor reading ---
    if sim_mode:
        reading = get_live_f2()
        using_arduino = False
    else:
        using_arduino = True
        if st.session_state.arduino_serial is None:
            st.session_state.arduino_serial = arduino_reader.open_arduino_serial()

        if st.session_state.arduino_serial is None:
            if st.session_state.last_valid_arduino_reading_f2 is not None:
                reading = st.session_state.last_valid_arduino_reading_f2
            else:
                st.warning("Arduino not found – using simulated data.")
                reading = get_live_f2()
                using_arduino = False
        else:
            result = arduino_reader.read_arduino_f2(st.session_state.arduino_serial)
            if result["ok"]:
                st.session_state.last_valid_arduino_reading_f2 = result["data"]
                st.session_state.consecutive_failures_f2 = 0
                reading = result["data"]
            else:
                st.session_state.consecutive_failures_f2 += 1
                if st.session_state.consecutive_failures_f2 > 10:
                    st.warning("No valid Arduino data for a while – using simulated data.")
                    reading = get_live_f2()
                    using_arduino = False
                else:
                    if st.session_state.last_valid_arduino_reading_f2 is not None:
                        reading = st.session_state.last_valid_arduino_reading_f2
                    else:
                        reading = get_live_f2()
                        using_arduino = False

    batch_df = st.session_state.f2_batch_data
    new_row = pd.DataFrame([{
        'day': batch_df['day'].max() + 0.1 if not batch_df.empty else 0.1,
        **reading
    }])
    st.session_state.f2_batch_data = pd.concat([batch_df, new_row], ignore_index=True)
    current_data = st.session_state.f2_batch_data

    if not current_data.empty:
        latest = current_data.iloc[-1]

        pressure_val = latest['pressure'] if latest['pressure'] is not None else np.nan
        water_val = latest['water_level'] if latest['water_level'] is not None else np.nan
        temp_val = latest['temperature'] if latest['temperature'] is not None else np.nan

        fpi, p_n, w_n = compute_f2_fpi(pressure_val, water_val, temp_val, current_data)
        D_max = 8
        days_left = days_remaining_f2(fpi, D_max, temp_val)
        progress = 100 * (1 - days_left / D_max) if D_max else 0

        # Main metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("FPI", f"{fpi:.2f}")
        col2.metric("Days Remaining", f"{days_left:.1f}")
        col3.metric("Progress", f"{progress:.0f}%")

        # Live sensor readings
        st.subheader("Live Sensor Readings")
        s1, s2, s3 = st.columns(3)
        s1.metric("Temperature", f"{temp_val:.1f} °C" if not np.isnan(temp_val) else "N/A")
        s2.metric("Water Level", f"{water_val:.1f} %" if not np.isnan(water_val) else "N/A")
        s3.metric("Pressure", f"{pressure_val:.2f} bar" if not np.isnan(pressure_val) else "N/A (no sensor yet)")

        status_text = "Live (Arduino)" if using_arduino else "Simulated"
        st.caption(f"Data source: {status_text}")

        alerts = f2_alerts(pressure_val, temp_val, water_val, fpi, days_left, latest['day'])
        if alerts:
            st.warning(" ⚠️ ".join(alerts))
        else:
            st.success("Carbonation proceeding normally.")
    else:
        st.info("No data yet.")

    # Plots
    st.subheader("Sensor Trends")
    if not current_data.empty:
        fig_p = px.line(current_data, x='day', y='pressure', title='Pressure (bar)')
        fig_t = px.line(current_data, x='day', y='temperature', title='Temperature (°C)')
        fig_w = px.line(current_data, x='day', y='water_level', title='Water Level (%)')
        st.plotly_chart(fig_p, use_container_width=True)
        st.plotly_chart(fig_t, use_container_width=True)
        st.plotly_chart(fig_w, use_container_width=True)

    # Validation
    st.subheader("Model Validation")
    if st.button("Validate F2 Model"):
        synthetic = pd.concat([generate_f2_batch(i) for i in range(5)], ignore_index=True)
        metrics, model = validate_f2_model(synthetic)
        st.dataframe(pd.DataFrame(metrics).T.style.format("{:.3f}"))

    if st.button("Log to History"):
        if 'history_rows' not in st.session_state:
            st.session_state.history_rows = []
        latest = current_data.iloc[-1].to_dict()
        latest['process'] = 'F2'
        latest['source'] = 'live' if sim_mode else 'live'
        st.session_state.history_rows.append(latest)
        st.success("Logged!")

    # Auto-refresh when using live Arduino data
    if not sim_mode:
        time.sleep(0.8)
        st.rerun()