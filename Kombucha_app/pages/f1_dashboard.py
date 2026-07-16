import sys
from pathlib import Path

# Allow importing arduino_reader from parent folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import time
import arduino_reader


# ------------------- Temperature efficiency & days remaining (F1) -------------------
def temp_efficiency(T, T_ideal=27, sigma=3.8):
    if T is None or (isinstance(T, float) and math.isnan(T)):
        T = T_ideal
    eff = math.exp(-(T - T_ideal)**2 / (2 * sigma**2))
    return 0.65 + 0.35 * eff


def days_remaining_from_fpi(fpi, D_max, T_current, T_ideal=27):
    if T_current is None or (isinstance(T_current, float) and math.isnan(T_current)):
        T_current = T_ideal
    temp_adjust = 1 + 0.02 * (T_ideal - T_current)
    return max(0, D_max * (1 - fpi) * temp_adjust)


# ------------------- Synthetic data generation -------------------
def generate_f1_batch(batch_id, days=12, ideal_temp=27, temp_std=1.0,
                      ph_start=4.1, ph_end=3.2, cond_start=200, cond_end=800,
                      turb_start=300, turb_end=50, color_start=80, color_end=60):
    np.random.seed(batch_id)
    t = np.linspace(0, days, 50)
    temp = ideal_temp + np.random.normal(0, temp_std, len(t))
    temp = np.clip(temp, 20, 35)
    
    ph = ph_start - (ph_start - ph_end) * (t / days)**0.8
    ph += np.random.normal(0, 0.03, len(t))
    cond = cond_start + (cond_end - cond_start) * (t / days)**0.9 + np.random.normal(0, 5, len(t))
    turb = turb_start - (turb_start - turb_end) * (t / days)**1.2 + np.random.normal(0, 8, len(t))
    color = color_start - (color_start - color_end) * (t / days)**0.5 + np.random.normal(0, 1, len(t))
    
    df = pd.DataFrame({
        'batch': batch_id,
        'day': t,
        'temperature': temp,
        'pH': np.clip(ph, 2.5, 5),
        'conductivity': np.clip(cond, 100, 1000),
        'turbidity': np.clip(turb, 10, 500),
        'color': np.clip(color, 30, 100)
    })
    return df


# ------------------- Live sensor simulation (fallback) -------------------
def get_live_f1():
    if 'f1_sim_state' not in st.session_state:
        st.session_state.f1_sim_state = {'day': 0, 'batch': 1}
    state = st.session_state.f1_sim_state
    t = state['day']
    ph = max(2.8, 4.1 - 0.08 * t + np.random.normal(0, 0.05))
    cond = 200 + 50 * t + np.random.normal(0, 10)
    turb = max(30, 300 - 25 * t + np.random.normal(0, 15))
    color = max(40, 80 - 2 * t + np.random.normal(0, 2))
    temp = 27 + np.random.normal(0, 1.5)
    water_lvl = 100 - 0.2*t + np.random.normal(0, 1)
    state['day'] += 0.1
    return {
        'temperature': temp,
        'pH': ph,
        'conductivity': cond,
        'turbidity': turb,
        'color': color,
        'water_level': water_lvl
    }


# ------------------- FPI calculation for F1 (with missing sensors) -------------------
def compute_f1_fpi(ph, cond, turb, color, temp, batch_data=None):
    # Fallback typical ranges
    ph_min, ph_max = 3.0, 4.5
    cond_min, cond_max = 100, 900
    turb_min, turb_max = 10, 500
    color_min, color_max = 30, 100

    if batch_data is not None and len(batch_data) >= 2:
        if not pd.isna(ph):
            ph_vals = batch_data['pH'].dropna()
            if len(ph_vals) > 0:
                ph_min, ph_max = ph_vals.min(), ph_vals.max()
        if not pd.isna(cond):
            c_vals = batch_data['conductivity'].dropna()
            if len(c_vals) > 0:
                cond_min, cond_max = c_vals.min(), c_vals.max()
        if not pd.isna(turb):
            t_vals = batch_data['turbidity'].dropna()
            if len(t_vals) > 0:
                turb_min, turb_max = t_vals.min(), t_vals.max()
        if not pd.isna(color):
            col_vals = batch_data['color'].dropna()
            if len(col_vals) > 0:
                color_min, color_max = col_vals.min(), col_vals.max()

    # Normalization with fallbacks
    if not pd.isna(ph) and ph_max != ph_min:
        ph_n = 1 - (ph - ph_min) / (ph_max - ph_min)
    else:
        ph_n = 1 - (3.8 - 3.0) / (4.5 - 3.0)

    if not pd.isna(cond) and cond_max != cond_min:
        cond_n = (cond - cond_min) / (cond_max - cond_min)
    else:
        cond_n = 0.5

    if not pd.isna(turb) and turb_max != turb_min:
        turb_n = 1 - (turb - turb_min) / (turb_max - turb_min)
    else:
        turb_n = 0.5

    if not pd.isna(color) and color_max != color_min:
        color_n = (color - color_min) / (color_max - color_min)
    else:
        color_n = 0.5

    ph_n = max(0, min(1, ph_n))
    cond_n = max(0, min(1, cond_n))
    turb_n = max(0, min(1, turb_n))
    color_n = max(0, min(1, color_n))

    w_ph, w_cond, w_turb, w_color = 0.38, 0.22, 0.18, 0.12
    raw_fpi = w_ph * ph_n + w_cond * cond_n + w_turb * turb_n + w_color * color_n
    temp_fac = temp_efficiency(temp)
    fpi = raw_fpi * temp_fac
    return fpi, ph_n, cond_n, turb_n, color_n


# ------------------- Alerts for F1 -------------------
def f1_alerts(ph, temp, conductivity, turbidity, fpi, days_remaining, day):
    alerts = []
    if ph is not None and not np.isnan(ph) and ph > 4.0 and day > 3:
        alerts.append("⚠️ pH still above 4.0 after 3 days – acidification may be slow.")
    if temp is not None and not np.isnan(temp) and temp < 25:
        alerts.append("❄️ Temperature below 25°C – fermentation slowed.")
    if temp is not None and not np.isnan(temp) and temp > 29:
        alerts.append("🔥 Temperature above 29°C – risk of off-flavours.")
    if days_remaining > 8 and fpi > 0.5:
        alerts.append("⏳ High FPI but long remaining days – check model calibration.")
    if conductivity is not None and not np.isnan(conductivity) and conductivity < 300 and day > 5:
        alerts.append("🔌 Conductivity low – weak metabolic activity?")
    return alerts


# ------------------- Model validation -------------------
def validate_f1_model(synthetic_data):
    df = synthetic_data.copy()
    D_max = 12
    df['actual_days_remaining'] = D_max - df['day']
    
    fpi_preds = []
    for _, row in df.iterrows():
        batch_data = df[df['batch'] == row['batch']] if 'batch' in df else None
        fpi, _, _, _, _ = compute_f1_fpi(
            row['pH'], row['conductivity'], row['turbidity'],
            row['color'], row['temperature'], batch_data
        )
        days_fpi = days_remaining_from_fpi(fpi, D_max, row['temperature'])
        fpi_preds.append(days_fpi)
    
    df['fpi_predicted_days'] = fpi_preds
    
    norm_ph = 1 - (df['pH'] - df['pH'].min()) / (df['pH'].max() - df['pH'].min())
    norm_cond = (df['conductivity'] - df['conductivity'].min()) / (df['conductivity'].max() - df['conductivity'].min())
    norm_turb = 1 - (df['turbidity'] - df['turbidity'].min()) / (df['turbidity'].max() - df['turbidity'].min())
    norm_color = (df['color'] - df['color'].min()) / (df['color'].max() - df['color'].min())
    temp_eff = df['temperature'].apply(temp_efficiency)
    
    X = pd.DataFrame({'ph_n': norm_ph, 'cond_n': norm_cond, 'turb_n': norm_turb,
                      'color_n': norm_color, 'temp_eff': temp_eff})
    y = df['actual_days_remaining']
    
    model = LinearRegression().fit(X, y)
    ml_preds = model.predict(X)
    
    rmse_fpi = math.sqrt(mean_squared_error(y, fpi_preds))
    mae_fpi = mean_absolute_error(y, fpi_preds)
    r2_fpi = r2_score(y, fpi_preds)
    
    rmse_ml = math.sqrt(mean_squared_error(y, ml_preds))
    mae_ml = mean_absolute_error(y, ml_preds)
    r2_ml = r2_score(y, ml_preds)
    
    metrics = {
        'FPI model': {'RMSE': rmse_fpi, 'MAE': mae_fpi, 'R²': r2_fpi},
        'ML model': {'RMSE': rmse_ml, 'MAE': mae_ml, 'R²': r2_ml}
    }
    return metrics, model


# ------------------- Streamlit page -------------------
def show():
    st.title("🍵 First Fermentation (F1) Monitor")
    
    if 'f1_batch_data' not in st.session_state:
        st.session_state.f1_batch_data = pd.DataFrame(
            columns=['day','temperature','pH','conductivity','turbidity','color','water_level']
        )
    if 'f1_batch_id' not in st.session_state:
        st.session_state.f1_batch_id = 1
    if 'arduino_serial' not in st.session_state:
        st.session_state.arduino_serial = None

    # For stable live mode: keep last valid Arduino reading and failure count
    if 'last_valid_arduino_reading_f1' not in st.session_state:
        st.session_state.last_valid_arduino_reading_f1 = None
    if 'consecutive_failures_f1' not in st.session_state:
        st.session_state.consecutive_failures_f1 = 0

    st.sidebar.header("F1 Controls")
    if st.sidebar.button("Start New Batch"):
        st.session_state.f1_batch_id += 1
        st.session_state.f1_batch_data = pd.DataFrame(
            columns=['day','temperature','pH','conductivity','turbidity','color','water_level']
        )
        st.session_state.f1_sim_state = {'day':0, 'batch': st.session_state.f1_batch_id}

    sim_mode = st.sidebar.checkbox("Use synthetic live data", value=True)

    # --- Get current sensor reading ---
    if sim_mode:
        reading = get_live_f1()
        using_arduino = False
    else:
        using_arduino = True
        if st.session_state.arduino_serial is None:
            st.session_state.arduino_serial = arduino_reader.open_arduino_serial()

        if st.session_state.arduino_serial is None:
            # No serial port at all → use last valid or simulate
            if st.session_state.last_valid_arduino_reading_f1 is not None:
                reading = st.session_state.last_valid_arduino_reading_f1
            else:
                st.warning("Arduino not found – using simulated data.")
                reading = get_live_f1()
                using_arduino = False
        else:
            result = arduino_reader.read_arduino_f1(st.session_state.arduino_serial)
            if result["ok"]:
                # Valid new reading
                st.session_state.last_valid_arduino_reading_f1 = result["data"]
                st.session_state.consecutive_failures_f1 = 0
                reading = result["data"]
            else:
                # No valid line this cycle
                st.session_state.consecutive_failures_f1 += 1
                if st.session_state.consecutive_failures_f1 > 10:
                    # Too many failures → fall back to simulation
                    st.warning("No valid Arduino data for a while – using simulated data.")
                    reading = get_live_f1()
                    using_arduino = False
                else:
                    # Keep using last known good reading
                    if st.session_state.last_valid_arduino_reading_f1 is not None:
                        reading = st.session_state.last_valid_arduino_reading_f1
                    else:
                        # Nothing yet → simulate temporarily
                        reading = get_live_f1()
                        using_arduino = False

    # --- Append to batch log ---
    batch_df = st.session_state.f1_batch_data
    new_row = pd.DataFrame([{
        'day': batch_df['day'].max() + 0.1 if not batch_df.empty else 0.1,
        **reading
    }])
    st.session_state.f1_batch_data = pd.concat([batch_df, new_row], ignore_index=True)
    current_data = st.session_state.f1_batch_data

    # --- Compute FPI and days remaining ---
    if not current_data.empty:
        latest = current_data.iloc[-1]

        ph_val = latest['pH'] if latest['pH'] is not None else np.nan
        cond_val = latest['conductivity'] if latest['conductivity'] is not None else np.nan
        turb_val = latest['turbidity'] if latest['turbidity'] is not None else np.nan
        color_val = latest['color'] if latest['color'] is not None else np.nan
        temp_val = latest['temperature'] if latest['temperature'] is not None else np.nan

        fpi, ph_n, cond_n, turb_n, color_n = compute_f1_fpi(
            ph_val, cond_val, turb_val, color_val, temp_val, current_data
        )
        D_max = 12
        days_left = days_remaining_from_fpi(fpi, D_max, temp_val)
        progress_pct = 100 * (1 - days_left / D_max) if D_max else 0

        # Main metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("FPI", f"{fpi:.2f}")
        col2.metric("Days Remaining", f"{days_left:.1f}")
        col3.metric("Progress", f"{progress_pct:.0f}%")

        # Live sensor readings
        st.subheader("Live Sensor Readings")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Temperature", f"{temp_val:.1f} °C" if not np.isnan(temp_val) else "N/A")
        s2.metric("pH", f"{ph_val:.2f}" if not np.isnan(ph_val) else "N/A")
        s3.metric("Conductivity", f"{cond_val:.0f} µS/cm" if not np.isnan(cond_val) else "N/A")
        s4.metric("Water Level", f"{latest['water_level']:.1f} %" if latest.get('water_level') is not None and not np.isnan(latest['water_level']) else "N/A")

        s5, s6 = st.columns(2)
        s5.metric("Turbidity", f"{turb_val:.1f} NTU" if not np.isnan(turb_val) else "N/A")
        s6.metric("Color Index", f"{color_val:.1f}" if not np.isnan(color_val) else "N/A")

        status_text = "Live (Arduino)" if using_arduino else "Simulated"
        st.caption(f"Data source: {status_text}")

        alerts = f1_alerts(ph_val, temp_val, cond_val, turb_val, fpi, days_left, latest['day'])
        if alerts:
            st.warning(" ⚠️ ".join(alerts))
        else:
            st.success("All systems normal.")
    else:
        st.info("Waiting for first reading...")

    # --- Sensor trend plots ---
    st.subheader("Sensor Trends (Current Batch)")
    if not current_data.empty:
        fig_ph = px.line(current_data, x='day', y='pH', title='pH')
        fig_cond = px.line(current_data, x='day', y='conductivity', title='Conductivity (µS/cm)')
        fig_turb = px.line(current_data, x='day', y='turbidity', title='Turbidity (NTU)')
        fig_color = px.line(current_data, x='day', y='color', title='Color Index')
        fig_temp = px.line(current_data, x='day', y='temperature', title='Temperature (°C)')
        fig_water = px.line(current_data, x='day', y='water_level', title='Water Level (%)')

        colA, colB = st.columns(2)
        with colA:
            st.plotly_chart(fig_ph, use_container_width=True)
            st.plotly_chart(fig_cond, use_container_width=True)
            st.plotly_chart(fig_temp, use_container_width=True)
        with colB:
            st.plotly_chart(fig_turb, use_container_width=True)
            st.plotly_chart(fig_color, use_container_width=True)
            st.plotly_chart(fig_water, use_container_width=True)

    # --- Model validation on synthetic data ---
    st.subheader("Model Validation (Synthetic Batches)")
    if st.button("Run Validation on 5 Synthetic Batches"):
        synthetic = pd.concat([generate_f1_batch(i) for i in range(5)], ignore_index=True)
        metrics, model = validate_f1_model(synthetic)
        df_metrics = pd.DataFrame(metrics).T
        st.dataframe(df_metrics.style.format("{:.3f}"))
        st.caption("FPI model = hand‑crafted formula; ML model = Linear Regression on normalized sensors")

    # --- Save to history (optional) ---
    if st.button("Log Current Reading to History"):
        if 'history_rows' not in st.session_state:
            st.session_state.history_rows = []
        latest = current_data.iloc[-1].to_dict()
        latest['process'] = 'F1'
        latest['source'] = 'live' if sim_mode else 'live'
        st.session_state.history_rows.append(latest)
        st.success("Logged!")

    # Auto-refresh when using live Arduino data
    if not sim_mode:
        time.sleep(0.8)
        st.rerun()