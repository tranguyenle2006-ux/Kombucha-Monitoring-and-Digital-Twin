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


# ------------------- Temperature efficiency & days remaining (F2) -------------------
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


# ------------------- FPI calculation for F2 (temperature only) -------------------
def compute_f2_fpi(temp, batch_data=None):
    """
    FPI based only on temperature profile vs time.
    Since we only have temperature, we treat FPI as proportional to elapsed time,
    adjusted by temperature efficiency.
    """
    if batch_data is None or len(batch_data) < 2:
        # No history yet; assume FPI grows linearly with day
        # We'll compute 'day' outside; here we just return a placeholder
        return 0.0, 0.0

    # Use day as proxy for progress, scaled by average temp efficiency
    day = batch_data['day'].max()
    D_max = 8  # typical F2 duration
    raw_progress = min(1.0, day / D_max)

    # Use current temp efficiency as modifier
    temp_fac = temp_efficiency(temp)
    # Normalize temp_fac to ~[0.65, 1.0]; treat 0.65 as 'neutral'
    fac_scaled = (temp_fac - 0.65) / (1.0 - 0.65)  # ~[0,1]
    fac_scaled = max(0, min(1, fac_scaled))

    fpi = raw_progress * (0.5 + 0.5 * fac_scaled)
    return fpi, raw_progress


# ------------------- Alerts for F2 -------------------
def f2_alerts(temp, fpi, days_left, day):
    alerts = []
    if temp is not None and not np.isnan(temp) and temp < 25:
        alerts.append("❄️ Temperature low – carbonation will be slow.")
    if temp is not None and not np.isnan(temp) and temp > 29:
        alerts.append("🔥 Temperature high – risk of off‑flavours and excessive pressure.")
    if days_left > 6 and fpi > 0.5:
        alerts.append("⏳ High FPI but long remaining days – check model calibration.")
    return alerts


# ------------------- Model validation (synthetic used only for internal checks) -------------------
def generate_f2_batch(batch_id, days=8, ideal_temp=27, temp_std=1.0):
    np.random.seed(batch_id)
    t = np.linspace(0, days, 40)
    temp = ideal_temp + np.random.normal(0, temp_std, len(t))
    temp = np.clip(temp, 20, 35)
    df = pd.DataFrame({
        'batch': batch_id,
        'day': t,
        'temperature': temp,
    })
    return df


def validate_f2_model(synthetic_data):
    D_max = 8
    df = synthetic_data.copy()
    df['actual_days_remaining'] = D_max - df['day']
    
    fpi_preds = []
    for _, row in df.iterrows():
        batch_data = df[df['batch'] == row['batch']]
        fpi, _ = compute_f2_fpi(row['temperature'], batch_data)
        days = days_remaining_f2(fpi, D_max, row['temperature'])
        fpi_preds.append(days)
    
    temp_eff = df['temperature'].apply(temp_efficiency)
    day_norm = df['day'] / D_max
    
    X = pd.DataFrame({'day_norm': day_norm, 'temp_eff': temp_eff})
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


# ------------------- Helper: get latest Arduino reading -------------------
def get_latest_arduino_reading():
    if 'arduino_serial' not in st.session_state or st.session_state.arduino_serial is None:
        st.session_state.arduino_serial = arduino_reader.open_arduino_serial()

    ser = st.session_state.arduino_serial
    if ser is None:
        return False, None

    result = arduino_reader.read_arduino_once(ser)
    if result["ok"] and result["temperature"] is not None:
        return True, result["temperature"]
    return False, None


# ------------------- Live sensors view (updated every second) -------------------
def show_live_sensors_f2():
    st.subheader("Live Sensors (F2)")

    placeholder = st.empty()
    stop_button = st.button("Stop Live View")

    if 'live_last_temp_f2' not in st.session_state:
        st.session_state.live_last_temp_f2 = None

    if stop_button:
        st.info("Live view stopped. Navigate to another tab or refresh to restart.")
        return

    while True:
        ok, temp = get_latest_arduino_reading()

        if ok:
            st.session_state.live_last_temp_f2 = temp

        temp = st.session_state.live_last_temp_f2

        with placeholder.container():
            if temp is None:
                st.warning("Waiting for Arduino data...")
            else:
                col1 = st.columns(1)[0]
                col1.metric("Temperature", f"{temp:.1f} °C" if temp is not None else "N/A")

                day_dummy = 0
                batch_dummy = pd.DataFrame({'day': [0], 'temperature': [temp]})
                fpi_dummy, _ = compute_f2_fpi(temp, batch_dummy) if temp is not None else (None, None)
                days_dummy = None
                if fpi_dummy is not None:
                    days_dummy = days_remaining_f2(fpi_dummy, 8, temp)

                if temp is not None:
                    alerts = f2_alerts(temp, fpi_dummy, days_dummy, day_dummy)
                    if alerts:
                        st.warning(" ⚠️ ".join(alerts))
                    else:
                        st.success("Carbonation conditions normal.")

        time.sleep(1)


# ------------------- Streamlit page -------------------
def show():
    st.title("🍾 Second Fermentation (F2) Monitor")

    # Session state init
    if 'f2_batch_data' not in st.session_state:
        st.session_state.f2_batch_data = pd.DataFrame(
            columns=['day','temperature']
        )
    if 'f2_batch_id' not in st.session_state:
        st.session_state.f2_batch_id = 1
    if 'arduino_serial' not in st.session_state:
        st.session_state.arduino_serial = None

    # Recording control (every 20 minutes)
    if 'f2_recording_on' not in st.session_state:
        st.session_state.f2_recording_on = False
    if 'f2_next_record_time' not in st.session_state:
        st.session_state.f2_next_record_time = None

    RECORD_INTERVAL_SEC = 20 * 60  # 20 minutes

    # Tabs: Batch & Graphs | Live Sensors
    tab_batch, tab_live = st.tabs(["Batch & Graphs", "Live Sensors"])

    with tab_batch:
        # Sidebar controls
        st.sidebar.header("F2 Controls")
        if st.sidebar.button("Start New Batch"):
            st.session_state.f2_batch_id += 1
            st.session_state.f2_batch_data = pd.DataFrame(
                columns=['day','temperature']
            )
            st.session_state.f2_next_record_time = None

        # Recording toggle button
        if st.sidebar.button(
            "Start Recording" if not st.session_state.f2_recording_on else "Stop Recording",
            key="f2_record_toggle"
        ):
            if not st.session_state.f2_recording_on:
                st.session_state.f2_recording_on = True
                st.session_state.f2_next_record_time = time.time()
            else:
                st.session_state.f2_recording_on = False
                st.session_state.f2_next_record_time = None

        # Status indicator
        if st.session_state.f2_recording_on:
            st.sidebar.success("✅ Recording ON (every 20 minutes)")
        else:
            st.sidebar.info("⏸️ Recording OFF")

        # Get latest Arduino reading
        ok, temp = get_latest_arduino_reading()

        if not ok:
            st.warning("No valid Arduino data at this moment. Waiting...")
            temp = None

        # Display live metric
        st.subheader("Current Sensor Readings")
        colT = st.columns(1)[0]
        colT.metric("Temperature", f"{temp:.1f} °C" if temp is not None else "Waiting for data")

        # Recording logic
        now = time.time()
        if st.session_state.f2_recording_on and temp is not None:
            if st.session_state.f2_next_record_time is None or now >= st.session_state.f2_next_record_time:
                batch_df = st.session_state.f2_batch_data
                new_day = batch_df['day'].max() + (RECORD_INTERVAL_SEC / 3600 / 24) if not batch_df.empty else 0

                new_row = pd.DataFrame([{
                    'day': new_day,
                    'temperature': temp
                }])
                st.session_state.f2_batch_data = pd.concat([batch_df, new_row], ignore_index=True)

                st.session_state.f2_next_record_time = now + RECORD_INTERVAL_SEC

        # Countdown timer
        if st.session_state.f2_recording_on and st.session_state.f2_next_record_time:
            remaining = max(0, st.session_state.f2_next_record_time - time.time())
            mins = int(remaining // 60)
            secs = int(remaining % 60)
            st.caption(f"Next recording in: {mins}m {secs}s")

        # Alerts
        if temp is not None:
            batch_df = st.session_state.f2_batch_data
            day_val = batch_df['day'].max() if not batch_df.empty else 0
            fpi, _ = compute_f2_fpi(temp, batch_df)
            days_left = days_remaining_f2(fpi, 8, temp)

            alerts = f2_alerts(temp, fpi, days_left, day_val)
            if alerts:
                st.warning(" ⚠️ ".join(alerts))
            else:
                st.success("Carbonation conditions normal.")

        # Batch & graphs
        st.subheader("Batch Data (Recorded Every 20 Minutes)")
        batch_df = st.session_state.f2_batch_data
        if not batch_df.empty:
            st.dataframe(batch_df, use_container_width=True)

            fig_temp = px.line(batch_df, x='day', y='temperature', title='Temperature Over Time (F2)')
            c1 = st.columns(1)[0]
            c1.plotly_chart(fig_temp, use_container_width=True)
        else:
            st.info("No recorded data yet. Turn recording ON and wait for the first 20‑minute interval.")

        # Optional model validation
        if st.button("Run Internal Model Validation (Synthetic Batches)"):
            synthetic = pd.concat([generate_f2_batch(i) for i in range(5)], ignore_index=True)
            metrics, model = validate_f2_model(synthetic)
            st.dataframe(pd.DataFrame(metrics).T.style.format("{:.3f}"))

    with tab_live:
        show_live_sensors_f2()

    # Save to history (optional)
    if st.button("Log Current Reading to History"):
        if 'history_rows' not in st.session_state:
            st.session_state.history_rows = []
        if temp is not None:
            row = {
                'process': 'F2',
                'source': 'live',
                'temperature': temp,
                'batch': st.session_state.f2_batch_id,
                'day': st.session_state.f2_batch_data['day'].max() if not st.session_state.f2_batch_data.empty else 0
            }
            st.session_state.history_rows.append(row)
            st.success("Logged!")