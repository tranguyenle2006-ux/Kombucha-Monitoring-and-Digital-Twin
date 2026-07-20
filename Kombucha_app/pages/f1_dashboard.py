import sys
from pathlib import Path

# Allow importing arduino_reader from parent folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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


# ------------------- FPI calculation for F1 (temperature + pH only) -------------------
def compute_f1_fpi(ph, temp, batch_data=None):
    """
    FPI based only on pH and temperature.
    No water level, no turbidity, no conductivity, no color.
    """
    # Fallback typical pH range
    ph_min, ph_max = 3.0, 4.5

    if batch_data is not None and len(batch_data) >= 2:
        if not pd.isna(ph):
            ph_vals = batch_data['pH'].dropna()
            if len(ph_vals) > 0:
                ph_min, ph_max = ph_vals.min(), ph_vals.max()

    if not pd.isna(ph) and ph_max != ph_min:
        ph_n = 1 - (ph - ph_min) / (ph_max - ph_min)
    else:
        ph_n = 1 - (3.8 - 3.0) / (4.5 - 3.0)

    ph_n = max(0, min(1, ph_n))

    # Only pH matters now; temperature is a modifier via temp_efficiency
    w_ph = 1.0
    raw_fpi = w_ph * ph_n
    temp_fac = temp_efficiency(temp)
    fpi = raw_fpi * temp_fac
    return fpi, ph_n


# ------------------- Alerts for F1 -------------------
def f1_alerts(ph, temp, fpi, days_remaining, day):
    alerts = []
    if ph is not None and not np.isnan(ph) and ph > 4.0 and day > 3:
        alerts.append("⚠️ pH still above 4.0 after 3 days – acidification may be slow.")
    if temp is not None and not np.isnan(temp) and temp < 25:
        alerts.append("❄️ Temperature below 25°C – fermentation slowed.")
    if temp is not None and not np.isnan(temp) and temp > 29:
        alerts.append("🔥 Temperature above 29°C – risk of off‑flavours.")
    if days_remaining > 8 and fpi > 0.5:
        alerts.append("⏳ High FPI but long remaining days – check model calibration.")
    return alerts


# ------------------- Model validation (synthetic used only for internal checks) -------------------
def generate_f1_batch(batch_id, days=12, ideal_temp=27, temp_std=1.0,
                      ph_start=4.1, ph_end=3.2):
    np.random.seed(batch_id)
    t = np.linspace(0, days, 50)
    temp = ideal_temp + np.random.normal(0, temp_std, len(t))
    temp = np.clip(temp, 20, 35)
    
    ph = ph_start - (ph_start - ph_end) * (t / days)**0.8
    ph += np.random.normal(0, 0.03, len(t))
    
    df = pd.DataFrame({
        'batch': batch_id,
        'day': t,
        'temperature': temp,
        'pH': np.clip(ph, 2.5, 5),
    })
    return df


def validate_f1_model(synthetic_data):
    df = synthetic_data.copy()
    D_max = 12
    df['actual_days_remaining'] = D_max - df['day']
    
    fpi_preds = []
    for _, row in df.iterrows():
        batch_data = df[df['batch'] == row['batch']] if 'batch' in df else None
        fpi, _ = compute_f1_fpi(row['pH'], row['temperature'], batch_data)
        days_fpi = days_remaining_from_fpi(fpi, D_max, row['temperature'])
        fpi_preds.append(days_fpi)
    
    df['fpi_predicted_days'] = fpi_preds
    
    norm_ph = 1 - (df['pH'] - df['pH'].min()) / (df['pH'].max() - df['pH'].min())
    temp_eff = df['temperature'].apply(temp_efficiency)
    
    X = pd.DataFrame({'ph_n': norm_ph, 'temp_eff': temp_eff})
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


# ------------------- Helper: get latest Arduino reading -------------------
def get_latest_arduino_reading():
    """
    Try to get a fresh Arduino reading.
    Returns (ok, temp, ph)
    """
    if 'arduino_serial' not in st.session_state or st.session_state.arduino_serial is None:
        st.session_state.arduino_serial = arduino_reader.open_arduino_serial()

    ser = st.session_state.arduino_serial
    if ser is None:
        return False, None, None

    result = arduino_reader.read_arduino_once(ser)
    if result["ok"]:
        return True, result["temperature"], result["pH"]
    return False, None, None


# ------------------- Live sensors view (updated every second) -------------------
def show_live_sensors_f1():
    st.subheader("Live Sensors (F1)")

    placeholder = st.empty()
    stop_button = st.button("Stop Live View")

    # Initialize persistent last values
    if 'live_last_temp_f1' not in st.session_state:
        st.session_state.live_last_temp_f1 = None
    if 'live_last_ph_f1' not in st.session_state:
        st.session_state.live_last_ph_f1 = None

    if stop_button:
        st.info("Live view stopped. Navigate to another tab or refresh to restart.")
        return

    while True:
        ok, temp, ph = get_latest_arduino_reading()

        if ok:
            st.session_state.live_last_temp_f1 = temp
            st.session_state.live_last_ph_f1 = ph

        temp = st.session_state.live_last_temp_f1
        ph = st.session_state.live_last_ph_f1

        with placeholder.container():
            if temp is None:
                st.warning("Waiting for Arduino data...")
            else:
                col1, col2 = st.columns(2)
                col1.metric("Temperature", f"{temp:.1f} °C" if temp is not None else "N/A")
                col2.metric("pH", f"{ph:.2f}" if ph is not None else "N/A")

                # Alerts
                day_dummy = 0  # live view doesn't track day
                fpi_dummy, _ = compute_f1_fpi(ph, temp, None) if temp is not None and ph is not None else (None, None)
                days_dummy = None
                if fpi_dummy is not None:
                    days_dummy = days_remaining_from_fpi(fpi_dummy, 12, temp)

                if temp is not None and ph is not None:
                    alerts = f1_alerts(ph, temp, fpi_dummy, days_dummy, day_dummy)
                    if alerts:
                        st.warning(" ⚠️ ".join(alerts))
                    else:
                        st.success("All systems normal.")

        time.sleep(1)
        # Rerun this live view by using st.rerun() would restart entire page;
        # instead, we loop here manually.


# ------------------- Streamlit page -------------------
def show():
    st.title("🍵 First Fermentation (F1) Monitor")

    # Session state init
    if 'f1_batch_data' not in st.session_state:
        st.session_state.f1_batch_data = pd.DataFrame(
            columns=['day','temperature','pH']
        )
    if 'f1_batch_id' not in st.session_state:
        st.session_state.f1_batch_id = 1
    if 'arduino_serial' not in st.session_state:
        st.session_state.arduino_serial = None

    # Recording control (every 20 minutes)
    if 'f1_recording_on' not in st.session_state:
        st.session_state.f1_recording_on = False
    if 'f1_next_record_time' not in st.session_state:
        st.session_state.f1_next_record_time = None
    if 'f1_last_recorded_temp' not in st.session_state:
        st.session_state.f1_last_recorded_temp = None
    if 'f1_last_recorded_ph' not in st.session_state:
        st.session_state.f1_last_recorded_ph = None

    RECORD_INTERVAL_SEC = 20 * 60  # 20 minutes

    # Tabs: Batch & Graphs | Live Sensors
    tab_batch, tab_live = st.tabs(["Batch & Graphs", "Live Sensors"])

    with tab_batch:
        # Sidebar controls
        st.sidebar.header("F1 Controls")
        if st.sidebar.button("Start New Batch"):
            st.session_state.f1_batch_id += 1
            st.session_state.f1_batch_data = pd.DataFrame(
                columns=['day','temperature','pH']
            )
            st.session_state.f1_next_record_time = None

        # Recording toggle button
        if st.sidebar.button(
            "Start Recording" if not st.session_state.f1_recording_on else "Stop Recording",
            key="f1_record_toggle"
        ):
            if not st.session_state.f1_recording_on:
                # Turn ON recording
                st.session_state.f1_recording_on = True
                st.session_state.f1_next_record_time = time.time()  # record immediately on next loop
            else:
                # Turn OFF recording
                st.session_state.f1_recording_on = False
                st.session_state.f1_next_record_time = None

        # Status indicator
        if st.session_state.f1_recording_on:
            st.sidebar.success("✅ Recording ON (every 20 minutes)")
        else:
            st.sidebar.info("⏸️ Recording OFF")

        # Get latest Arduino reading (for display and possible recording)
        ok, temp, ph = get_latest_arduino_reading()

        if not ok:
            st.warning("No valid Arduino data at this moment. Waiting...")
            temp = None
            ph = None

        # Display live metrics
        st.subheader("Current Sensor Readings")
        colT, colP = st.columns(2)
        colT.metric("Temperature", f"{temp:.1f} °C" if temp is not None else "Waiting for data")
        colP.metric("pH", f"{ph:.2f}" if ph is not None else "Waiting for data")

        # Recording logic
        now = time.time()
        if st.session_state.f1_recording_on and temp is not None and ph is not None:
            if st.session_state.f1_next_record_time is None or now >= st.session_state.f1_next_record_time:
                # Record a new point
                batch_df = st.session_state.f1_batch_data
                new_day = batch_df['day'].max() + (RECORD_INTERVAL_SEC / 3600 / 24) if not batch_df.empty else 0

                new_row = pd.DataFrame([{
                    'day': new_day,
                    'temperature': temp,
                    'pH': ph
                }])
                st.session_state.f1_batch_data = pd.concat([batch_df, new_row], ignore_index=True)

                st.session_state.f1_last_recorded_temp = temp
                st.session_state.f1_last_recorded_ph = ph
                st.session_state.f1_next_record_time = now + RECORD_INTERVAL_SEC

        # Countdown timer
        if st.session_state.f1_recording_on and st.session_state.f1_next_record_time:
            remaining = max(0, st.session_state.f1_next_record_time - time.time())
            mins = int(remaining // 60)
            secs = int(remaining % 60)
            st.caption(f"Next recording in: {mins}m {secs}s")

        # Alerts
        if temp is not None and ph is not None:
            # Use latest point for "day"
            batch_df = st.session_state.f1_batch_data
            day_val = batch_df['day'].max() if not batch_df.empty else 0
            fpi, ph_n = compute_f1_fpi(ph, temp, batch_df)
            days_left = days_remaining_from_fpi(fpi, 12, temp)

            alerts = f1_alerts(ph, temp, fpi, days_left, day_val)
            if alerts:
                st.warning(" ⚠️ ".join(alerts))
            else:
                st.success("All systems normal.")

        # Batch & graphs
        st.subheader("Batch Data (Recorded Every 20 Minutes)")
        batch_df = st.session_state.f1_batch_data
        if not batch_df.empty:
            st.dataframe(batch_df, use_container_width=True)

            fig_temp = px.line(batch_df, x='day', y='temperature', title='Temperature Over Time (F1)')
            fig_ph = px.line(batch_df, x='day', y='pH', title='pH Over Time (F1)')

            c1, c2 = st.columns(2)
            c1.plotly_chart(fig_temp, use_container_width=True)
            c2.plotly_chart(fig_ph, use_container_width=True)
        else:
            st.info("No recorded data yet. Turn recording ON and wait for the first 20‑minute interval.")

        # Optional model validation button (internal/synthetic only)
        if st.button("Run Internal Model Validation (Synthetic Batches)"):
            synthetic = pd.concat([generate_f1_batch(i) for i in range(5)], ignore_index=True)
            metrics, model = validate_f1_model(synthetic)
            df_metrics = pd.DataFrame(metrics).T
            st.dataframe(df_metrics.style.format("{:.3f}"))
            st.caption("FPI model = hand‑crafted (pH + temp); ML model = Linear Regression")

    with tab_live:
        show_live_sensors_f1()

    # Save to history (optional)
    if st.button("Log Current Reading to History"):
        if 'history_rows' not in st.session_state:
            st.session_state.history_rows = []
        if temp is not None or ph is not None:
            row = {
                'process': 'F1',
                'source': 'live',
                'temperature': temp,
                'pH': ph,
                'batch': st.session_state.f1_batch_id,
                'day': st.session_state.f1_batch_data['day'].max() if not st.session_state.f1_batch_data.empty else 0
            }
            st.session_state.history_rows.append(row)
            st.success("Logged!")  