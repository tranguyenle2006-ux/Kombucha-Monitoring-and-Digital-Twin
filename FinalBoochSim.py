import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


st.set_page_config(layout="wide", page_title="Kombucha Simulation")
st.title("Kombucha Fermentation Simulation & Estimation Dashboard")


class KombuchaStateEstimator:
    def __init__(self, S0, V_vessel, EC0, alpha=0.12, beta=0.05, m1=0.85, m2=-0.15):
        self.S0 = S0
        self.V_vessel = V_vessel
        self.V_headspace = V_vessel * 0.08
        self.V_liquid_t0 = V_vessel * 0.92
        self.EC0 = EC0
        self.prev_ec = EC0
        self.alpha = alpha
        self.beta = beta
        self.m1 = m1
        self.m2 = m2
        self.S_t = S0
        self.X_y = 0.05
        self.X_b = 0.02
        self.acetic_acid = 0.0
        self.gluconic_acid = 0.0

    def estimate_step(self, sensor_row, is_f2=False):
        ph = sensor_row["pH"]
        ec = sensor_row["EC"]
        ntu = sensor_row["NTU"]
        abs_val = sensor_row["Absorbance"]
        temp = sensor_row["Temp_C"]
        pressure_bar = sensor_row["Pressure_bar"]

        pressure_atm = 1.0 + (pressure_bar * 0.9869232667)
        temp_factor = np.exp((temp - 23.0) / 10.0)

        yeast_growth_potential = max(0.0, (ntu * 0.001) * self.m1)
        bacteria_growth_potential = max(0.0, (abs_val * 0.05) + (self.m2 * 0.1))
        self.X_y += yeast_growth_potential * temp_factor
        self.X_b += bacteria_growth_potential * temp_factor

        consumption_rate = (0.012 * self.X_y + 0.005 * self.X_b) * temp_factor
        self.S_t = max(5.0, self.S_t - consumption_rate)

        delta_ec_step = max(0.0, ec - self.prev_ec)
        self.prev_ec = ec
        new_acid_step = delta_ec_step * 12.0 * self.alpha * temp_factor
        ph_factor = max(0.1, (4.5 - ph) * self.beta)
        acetic_ratio = min(0.9, 0.4 + ph_factor)

        self.acetic_acid += new_acid_step * acetic_ratio
        self.gluconic_acid += new_acid_step * (1.0 - acetic_ratio)

        co2_moles = 0.0
        ethanol_moles = 0.0
        if is_f2:
            temp_k = temp + 273.15
            kh_std = 0.034
            kh_t = kh_std * np.exp(2400.0 * ((1.0 / temp_k) - (1.0 / 298.15)))
            co2_dissolved_moles = kh_t * pressure_atm * self.V_liquid_t0
            co2_gas_moles = (pressure_atm * self.V_headspace) / (0.0821 * temp_k)
            co2_moles = co2_gas_moles + co2_dissolved_moles
            ethanol_moles = co2_moles * 0.92

        return {
            "Sugar_Remaining_g_L": self.S_t,
            "Biomass_Yeast_g_L": self.X_y,
            "Biomass_Bacteria_g_L": self.X_b,
            "Acetic_Acid_g_L": self.acetic_acid,
            "Gluconic_Acid_g_L": self.gluconic_acid,
            "CO2_Total_Moles": co2_moles,
            "Ethanol_Created_Moles": ethanol_moles,
            "Kinetic_Temp_Factor": temp_factor,
        }


def load_sensor_telemetry():
    total_days = 14
    timesteps = total_days * 24
    hours = np.arange(timesteps)
    days = hours / 24.0

    f1_end_day = 11.0
    fridge_start_day = 12.5

    temp_curve = []
    for d in days:
        if d < fridge_start_day:
            temp_curve.append(22.5 + 4.5 * np.sin(2 * np.pi * d))
        else:
            hours_in_fridge = (d - fridge_start_day) * 24.0
            temp_curve.append(3.0 + (22.5 - 3.0) * np.exp(-0.25 * hours_in_fridge))
    temp_curve = np.array(temp_curve)

    temp_factors = np.exp((temp_curve - 23.0) / 10.0)

    pressure_curve = np.zeros(timesteps)
    for idx, d in enumerate(days):
        if d < f1_end_day:
            pressure_curve[idx] = 0.0
        elif d < fridge_start_day:
            fraction_f2 = (d - f1_end_day) / (fridge_start_day - f1_end_day)
            pressure_curve[idx] = (fraction_f2 * 0.18 * temp_factors[idx]) + np.random.normal(0, 0.002)
        else:
            hours_in_fridge = (d - fridge_start_day) * 24.0
            starting_p = pressure_curve[int(fridge_start_day * 24) - 1]
            target_p = 0.05
            pressure_curve[idx] = target_p + (starting_p - target_p) * np.exp(-0.18 * hours_in_fridge) + np.random.normal(0, 0.002)
    pressure_curve = np.clip(pressure_curve, 0.0, None)

    ec_curve = []
    current_ec = 2.0
    for idx, d in enumerate(days):
        rate = 0.01 * temp_factors[idx]
        current_ec = min(5.5, current_ec + rate)
        ec_curve.append(current_ec + np.random.normal(0, 0.01))
    ec_curve = np.array(ec_curve)

    turbidity_curve = []
    current_turb = 10.0
    for idx, d in enumerate(days):
        growth = (0.8 * temp_factors[idx]) if d < 8.0 else (-0.3 * temp_factors[idx])
        current_turb = max(5.0, current_turb + growth)
        turbidity_curve.append(current_turb + np.random.normal(0, 0.5))
    turbidity_curve = np.array(turbidity_curve)

    ph_curve = []
    current_ph = 4.2
    for idx, d in enumerate(days):
        decay_rate = 0.005 * temp_factors[idx]
        current_ph = max(3.0, current_ph - (current_ph - 3.0) * decay_rate)
        ph_curve.append(current_ph + np.random.normal(0, 0.005))
    ph_curve = np.array(ph_curve)

    abs_curve = np.linspace(0.1, 0.7, timesteps) + np.random.normal(0, 0.01, timesteps)
    vol_curve = np.linspace(3.8, 3.55, timesteps)

    df = pd.DataFrame({
        "Day": days,
        "pH": ph_curve,
        "EC": ec_curve,
        "NTU": turbidity_curve,
        "Absorbance": abs_curve,
        "Temp_C": temp_curve,
        "Pressure_bar": pressure_curve
    })
    return df


def make_line_chart(df, x_col, y_cols, y_label, show_f2_line=True, f2_day=11.0, fridge_day=12.5):
    fig = go.Figure()
    if isinstance(y_cols, str):
        y_cols = [y_cols]
    for col in y_cols:
        fig.add_trace(go.Scatter(x=df[x_col], y=df[col], mode='lines', name=col))

    if show_f2_line:
        fig.add_vline(
            x=f2_day,
            line_width=2,
            line_dash="dash",
            line_color="rgba(255, 75, 75, 0.9)",
            annotation_text="F2 Start",
            annotation_position="top left"
        )
        fig.add_vline(
            x=fridge_day,
            line_width=2,
            line_dash="dash",
            line_color="rgba(75, 150, 255, 0.9)",
            annotation_text="To Fridge",
            annotation_position="top left"
        )

    fig.update_layout(
        xaxis_title="Timeline (Days)",
        yaxis_title=y_label,
        margin=dict(l=10, r=10, t=10, b=10),
        height=260,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


st.sidebar.header("Physical Vessel & Batch Setup")
vessel_volume = st.sidebar.slider("Total Vessel Volume (L)", 1.0, 10.0, 4.0, step=0.5)
S0_input = st.sidebar.slider("Starting Sugar (S0) [g/L]", 50.0, 100.0, 80.0, step=5.0)

calculated_headspace = vessel_volume * 0.08
calculated_liquid = vessel_volume * 0.92

st.sidebar.markdown(f"""
**Determined Physical Volumes:**
* **Headspace (8% of Vessel):** `{calculated_headspace:.3f} L`
* **Target Liquid Volume (92%):** `{calculated_liquid:.3f} L`
""")

st.sidebar.header("Global Tuning Parameters")
alpha = st.sidebar.slider("alpha (Acid Scaling Coefficient)", 0.05, 0.25, 0.12)
beta = st.sidebar.slider("beta (pH Sensitivity Index)", 0.01, 0.10, 0.05)
m1 = st.sidebar.slider("m1 (Yeast Growth Multiplier)", 0.5, 1.2, 0.85)
m2 = st.sidebar.slider("m2 (Bacteria Base Offset)", -0.3, 0.0, -0.15)

df_sensors = load_sensor_telemetry()

estimator = KombuchaStateEstimator(
    S0=S0_input, V_vessel=vessel_volume, EC0=2.0,
    alpha=alpha, beta=beta, m1=m1, m2=m2
)

estimations = []
f2_boundary_day = 11.0
for _, row in df_sensors.iterrows():
    is_f2_phase = row["Day"] >= f2_boundary_day
    estimations.append(estimator.estimate_step(row, is_f2=is_f2_phase))

df_estimates = pd.DataFrame(estimations)
df_estimates["Day"] = df_sensors["Day"]

tab1, tab2 = st.tabs(["Raw Sensor Signals", "Biological Estimations"])

with tab1:
    st.subheader("Raw Physical Sensor Signals (Observed States)")

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        with st.container(border=True):
            st.markdown("#### Headspace Pressure (bar)")
            st.plotly_chart(make_line_chart(df_sensors, "Day", "Pressure_bar", "bar"), width="stretch")
            st.caption("F2 pressure accumulation is accelerated by heat, then drops sharply upon refrigeration due to increased liquid solubility.")

    with row1_col2:
        with st.container(border=True):
            st.markdown("#### Turbidity (NTU)")
            st.plotly_chart(make_line_chart(df_sensors, "Day", "NTU", "NTU"), width="stretch")
            st.caption("Yeast concentration. Notice the faster upward slopes during warm cycles.")

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        with st.container(border=True):
            st.markdown("#### Thermal Profile (°C)")
            st.plotly_chart(make_line_chart(df_sensors, "Day", "Temp_C", "°C"), width="stretch")
            st.caption("Ambient diurnal room temperature fluctuations shifting into rapid fridge cooling.")

    with row2_col2:
        with st.container(border=True):
            st.markdown("#### Electrical Conductivity (mS/cm)")
            st.plotly_chart(make_line_chart(df_sensors, "Day", "EC", "mS/cm"), width="stretch")
            st.caption("Ionic progression increases step-wise during peak daily temperature periods.")

    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        with st.container(border=True):
            st.markdown("#### Media pH")
            st.plotly_chart(make_line_chart(df_sensors, "Day", "pH", "pH"), width="stretch")
            st.caption("pH drops in stair-steps matching warm kinetic windows when bacteria are highly active.")

    with row3_col2:
        with st.container(border=True):
            st.markdown("#### Optical Density / Absorbance")
            st.plotly_chart(make_line_chart(df_sensors, "Day", "Absorbance", "OD"), width="stretch")

with tab2:
    st.subheader("Biological & Chemical Estimations (Hidden States)")

    with st.container(border=True):
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Vessel Volume Config", f"{vessel_volume:.2f} L")
        with m_col2:
            st.metric("Headspace Target (8%)", f"{calculated_headspace:.3f} L")
        with m_col3:
            st.metric("Current Sim Batch S0", f"{S0_input} g/L")

    with st.container(border=True):
        col_g, col_desc = st.columns([1.2, 0.8])
        with col_g:
            st.markdown("#### Arrhenius Kinetic Factor (k_T)")
            st.plotly_chart(make_line_chart(df_estimates, "Day", "Kinetic_Temp_Factor", "Multiplier (x)"), width="stretch")
        with col_desc:
            st.markdown("##### The Metabolic Engine")
            st.write("This factor scales reaction kinetics. When warm, metabolism speeds up. Chilling to 3°C halts activity (k_T ≈ 0.15), pausing growth and acid development.")

    with st.container(border=True):
        col_g, col_desc = st.columns([1.2, 0.8])
        with col_g:
            st.markdown("#### Sugar Depletion Curve")
            st.plotly_chart(make_line_chart(df_estimates, "Day", "Sugar_Remaining_g_L", "Sugar (g/L)"), width="stretch")
        with col_desc:
            st.markdown("##### Dynamic Sugar Decay Model")
            with st.expander("Show Coupled Equations", expanded=True):
                st.latex(r"k_T = e^{\frac{T - 23.0}{10.0}}")
                st.latex(r"\frac{dS}{dt} = -k_T \cdot \left(0.012 \cdot X_y + 0.005 \cdot X_b\right)")

    with st.container(border=True):
        col_g, col_desc = st.columns([1.2, 0.8])
        with col_g:
            st.markdown("#### Cumulative Yeast vs. Bacteria Biomass")
            st.plotly_chart(make_line_chart(df_estimates, "Day", ["Biomass_Yeast_g_L", "Biomass_Bacteria_g_L"], "Biomass (g/L)"), width="stretch")
        with col_desc:
            st.markdown("##### Biologically Correct Integration")
            st.write("Growth rates slow down during cooler cycles (showing plateaus), but existing cell mass is preserved rather than destroyed.")
            with st.expander("Show Dynamic Integral Equations", expanded=True):
                st.latex(r"X_{y,t} = X_{y,t-1} + \max(0, \text{NTU} \times 0.001 \times m_1) \times k_T")
                st.latex(r"X_{b,t} = X_{b,t-1} + \max(0, \text{Abs} \times 0.05 + m_2 \times 0.1) \times k_T")

    with st.container(border=True):
        col_g, col_desc = st.columns([1.2, 0.8])
        with col_g:
            st.markdown("#### Cumulative Coupled Organic Acid Growth")
            st.plotly_chart(make_line_chart(df_estimates, "Day", ["Acetic_Acid_g_L", "Gluconic_Acid_g_L"], "Acidity (g/L)"), width="stretch")
        with col_desc:
            st.markdown("##### Biologically Accurate Non-Decreasing Acids")
            st.write("Acids are modeled as stable accumulated chemical products. Growth rate plateaus dynamically when cold, but the accumulated chemical yield remains flat and preserved.")
            with st.expander("Show Equations", expanded=True):
                st.latex(r"\Delta\text{Acid}_{\text{step}} = (EC_t - EC_{t-1}) \times 12 \times \alpha \times k_T")
                st.latex(r"\text{Acetic}_t = \text{Acetic}_{t-1} + \Delta\text{Acid}_{\text{step}} \times \text{Ratio}_{\text{Acetic}}")

    with st.container(border=True):
        col_g, col_desc = st.columns([1.2, 0.8])
        with col_g:
            st.markdown("#### F2 Carbonation & Alcohol Accumulation")
            st.plotly_chart(make_line_chart(df_estimates, "Day", ["CO2_Total_Moles", "Ethanol_Created_Moles"], "Moles"), width="stretch")
        with col_desc:
            st.markdown("##### Henry's Law Solubility Drops")
            st.markdown(f"**Liquid phase volume ($V_L$):** `{calculated_liquid:.3f} L` | **Headspace ($V_H$):** `{calculated_headspace:.3f} L`")
            with st.expander("Show Equations", expanded=True):
                st.latex(r"K_H(T) = K_{H,\text{std}} \times e^{2400 \times \left[ \frac{1}{T} - \frac{1}{298.15} \right]}")
                st.latex(r"P_{\text{atm}} = 1 + 0.986923 \cdot P_{\text{bar}}")
                st.latex(r"CO_2 = CO_{2,\text{gas}} + CO_{2,\text{dissolved}}")