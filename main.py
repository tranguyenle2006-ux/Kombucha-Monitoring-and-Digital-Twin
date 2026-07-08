import streamlit as st
import serial
import pandas as pd
import plotly.express as px
from datetime import datetime

PORT = "COM4"      # Change if needed
BAUD = 9600

st.set_page_config(layout="wide")
st.title("Kombucha Digital Twin")

graph1 = st.empty()
graph2 = st.empty()
table = st.empty()

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(
        columns=["DateTime","WaterLevel","Temperature"]
    )

try:
    ser = serial.Serial(PORT, BAUD, timeout=1)

    while True:

        line = ser.readline().decode().strip()

        if line == "":
            continue

        try:

            date,time,water,temp = line.split(",")

            dt = datetime.strptime(
                date+" "+time,
                "%Y-%m-%d %H:%M:%S"
            )

            row = pd.DataFrame(
                [[dt,int(water),float(temp)]],
                columns=["DateTime","WaterLevel","Temperature"]
            )

            st.session_state.df = pd.concat(
                [st.session_state.df,row],
                ignore_index=True
            )

            fig1 = px.line(
                st.session_state.df,
                x="DateTime",
                y="WaterLevel",
                markers=True,
                title="Water Level"
            )

            fig2 = px.line(
                st.session_state.df,
                x="DateTime",
                y="Temperature",
                markers=True,
                title="Temperature (°C)"
            )

            graph1.plotly_chart(fig1, use_container_width=True)
            graph2.plotly_chart(fig2, use_container_width=True)

            table.dataframe(
                st.session_state.df.tail(10),
                use_container_width=True
            )

        except:
            pass

except Exception as e:
    st.error(e)