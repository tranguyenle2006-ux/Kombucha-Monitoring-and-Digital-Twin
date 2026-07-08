import streamlit as st
import serial
import pandas as pd
import plotly.express as px
from datetime import datetime
import time


PORT="COM4"
BAUD=9600


st.set_page_config(
    page_title="Kombucha Digital Twin",
    layout="wide"
)


st.title("🍵 Kombucha Digital Twin")



# --------------------
# Connect Arduino
# --------------------

if "ser" not in st.session_state:

    st.session_state.ser = serial.Serial(
        PORT,
        BAUD,
        timeout=1
    )



ser = st.session_state.ser



# --------------------
# Data storage
# --------------------

if "data" not in st.session_state:

    st.session_state.data = pd.DataFrame(
        columns=[
            "Time",
            "Water Level",
            "Temperature"
        ]
    )



# --------------------
# Sidebar Control
# --------------------

st.sidebar.header(
    "Sampling Control"
)


interval = st.sidebar.selectbox(
    "Collect data every:",
    [
        1,
        5,
        15,
        30,
        60
    ]
)



if st.sidebar.button("Apply Interval"):


    command = f"SET{interval}\n"


    ser.write(
        command.encode()
    )


    st.sidebar.success(
        f"Set to every {interval} minutes"
    )



# --------------------
# Read Arduino
# --------------------


if ser.in_waiting:


    line = ser.readline().decode().strip()


    try:

        date,timeStamp,water,temp = line.split(",")


        dt=datetime.strptime(
            date+" "+timeStamp,
            "%Y-%m-%d %H:%M:%S"
        )


        new=pd.DataFrame(
            [[
                dt,
                int(water),
                float(temp)
            ]],
            columns=[
                "Time",
                "Water Level",
                "Temperature"
            ]
        )


        st.session_state.data=pd.concat(
            [
                st.session_state.data,
                new
            ],
            ignore_index=True
        )


    except:
        pass




# --------------------
# Display
# --------------------

df=st.session_state.data



if len(df)>0:


    c1,c2,c3=st.columns(3)


    c1.metric(
        "Temperature",
        f"{df.iloc[-1]['Temperature']} °C"
    )


    c2.metric(
        "Water Level",
        df.iloc[-1]["Water Level"]
    )


    c3.metric(
        "Samples",
        len(df)
    )



    fig=px.line(
        df,
        x="Time",
        y=[
            "Temperature",
            "Water Level"
        ],
        markers=True
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


else:

    st.info(
        "Waiting for data..."
    )



time.sleep(1)

st.rerun()