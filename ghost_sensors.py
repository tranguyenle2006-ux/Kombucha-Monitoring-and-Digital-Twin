"""
Ghost Sensors

These simulate the real sensors before
the hardware arrives.
"""

import numpy as np

from .base_sensor import Sensor


class GhostPHSensor(Sensor):
    def __init__(self):
        super().__init__("pH", "pH")
        self.initial = 4.30
        self.final = 2.75
        self.k = 0.33

    def read(self, day):
        value = self.final + (self.initial - self.final) * np.exp(-self.k * day)
        noise = np.random.normal(0, 0.02)
        return round(value + noise, 3)


class GhostConductivitySensor(Sensor):
    def __init__(self):
        super().__init__("Conductivity", "mS/cm")
        self.start = 2.0
        self.end = 5.2

    def read(self, day):
        value = self.start + (self.end - self.start) * (1 - np.exp(-0.40 * day))
        noise = np.random.normal(0, 0.05)
        return round(value + noise, 2)


class GhostTemperatureSensor(Sensor):
    def __init__(self):
        super().__init__("Temperature", "°C")

    def read(self, hour):
        value = 23 + 2 * np.sin(hour / 24 * 2 * np.pi)
        noise = np.random.normal(0, 0.2)
        return round(value + noise, 2)


class GhostTurbiditySensor(Sensor):
    def __init__(self):
        super().__init__("Turbidity", "NTU")

    def read(self, day):
        value = 150 * np.exp(-(day - 4) ** 2 / 5)
        noise = np.random.normal(0, 5)
        return max(0, round(value + noise, 1))


class GhostWaterLevelSensor(Sensor):
    def __init__(self):
        super().__init__("Water Level", "%")

    def read(self, day):
        value = 100 - day * 0.6
        noise = np.random.normal(0, 0.15)
        return round(value + noise, 2)


class GhostPressureSensor(Sensor):
    def __init__(self):
        super().__init__("Pressure", "psi")
        self.maximum = 30

    def read(self, day):
        value = self.maximum * (1 - np.exp(-0.8 * day))
        noise = np.random.normal(0, 0.3)
        return round(value + noise, 2)