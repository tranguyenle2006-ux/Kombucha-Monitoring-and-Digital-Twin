"""
Core mathematical model for the kombucha digital twin.
This file contains ONLY the process equations.
"""

import numpy as np

PARAMETERS = {

    "k_sugar":0.018,
    "mu_yeast":0.42,
    "mu_bacteria":0.26,

    "Ks_yeast":18,
    "Ks_bacteria":10,

    "kd_yeast":0.010,
    "kd_bacteria":0.008,

    "yield_ethanol":0.46,
    "yield_acid":0.65,
    "yield_co2":0.32,

    "theta":1.05,
    "reference_temperature":25
}

def temperature_factor(temp):

    theta = PARAMETERS["theta"]
    ref = PARAMETERS["reference_temperature"]

    return theta**(temp-ref)

def fermentation_odes(t,
                      state,
                      temperature):

    """
    State Vector
    state[0] = Sugar
    state[1] = Yeast
    state[2] = Bacteria
    state[3] = Ethanol
    state[4] = Acetic Acid
    state[5] = Dissolved CO2
    """

    S = state[0]
    Xy = state[1]
    Xb = state[2]
    E = state[3]
    A = state[4]
    C = state[5]

    T = temperature_factor(
        temperature
    )

    mu_y = PARAMETERS["mu_yeast"] * T
    mu_b = PARAMETERS["mu_bacteria"] * T

    Ks_y = PARAMETERS["Ks_yeast"]
    Ks_b = PARAMETERS["Ks_bacteria"]

    kd_y = PARAMETERS["kd_yeast"]
    kd_b = PARAMETERS["kd_bacteria"]


    yeast_growth = (
        mu_y *
        S/(Ks_y+S) *
        Xy
    )

    bacteria_growth = (
        mu_b *
        E/(Ks_b+E) *
        Xb
    )

    dS = -PARAMETERS["k_sugar"] * S * Xy
    dXy = yeast_growth - kd_y*Xy
    dXb = bacteria_growth - kd_b*Xb

    ethanol_created = (
        PARAMETERS["yield_ethanol"]
        * (-dS)
    )

    ethanol_used = (
        0.45
        * bacteria_growth
    )

    dE = ethanol_created - ethanol_used
    dA = (
        PARAMETERS["yield_acid"]
        * ethanol_used
    )

    dC = (
        PARAMETERS["yield_co2"]
        * ethanol_created
    )

    return [
        dS,
        dXy,
        dXb,
        dE,
        dA,
        dC
    ]

def estimate_pH(acid):

    pH = 4.30 - 0.055*acid
    return max(2.5,pH)

def estimate_conductivity(acid):
    return 2.0 + 0.085*acid

def estimate_turbidity(yeast,
                       bacteria):

    biomass = yeast+bacteria
    return biomass*120

def estimate_pressure(co2):
    return co2*4.8

def estimate_water_level(day):
    return 100-0.60*day