import serial
import time
from serial.tools import list_ports
import re

BAUD_RATE = 9600

def find_arduino_port():
    ports = list_ports.comports()
    candidates = [
        p.device for p in ports
        if "Arduino" in p.description
        or "usbmodem" in p.device.lower()
        or "usbserial" in p.device.lower()
    ]
    return candidates[0] if candidates else None

def open_arduino_serial():
    port = find_arduino_port()
    if port is None:
        return None
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=0.1)
        time.sleep(2)  # give Arduino time to reset
        return ser
    except Exception:
        return None

def parse_arduino_line(line):
    """
    Expected format from your Arduino:
      Temperature: 26.80 C, Water Level: 74 %, pH: 4.21

    Returns:
      {
        "temperature": float or None,
        "water_level": float or None,
        "pH": float or None
      }
    """
    data = {}

    t_match = re.search(r"Temperature:\s*([-+]?\d*\.?\d+)", line)
    if t_match:
        data["temperature"] = float(t_match.group(1))

    w_match = re.search(r"Water Level:\s*([-+]?\d*\.?\d+)", line)
    if w_match:
        data["water_level"] = float(w_match.group(1))

    ph_match = re.search(r"pH:\s*([-+]?\d*\.?\d+)", line)
    if ph_match:
        data["pH"] = float(ph_match.group(1))

    return data


def read_arduino_f1(ser):
    """
    For F1: use temperature, pH; water_level is optional.
    Other sensors are None for now.

    Returns:
      {
        "ok": True/False,
        "data": {
            "temperature": float or None,
            "pH": float or None,
            "water_level": float or None,
            "conductivity": None,
            "turbidity": None,
            "color": None,
        } or None
      }

    ok=False means: no valid line read yet (empty or unparseable).
    """
    if ser is None or not ser.is_open:
        return {"ok": False, "data": None}

    try:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            # No complete line yet; not necessarily an error
            return {"ok": False, "data": None}

        parsed = parse_arduino_line(line)

        # Consider it valid only if we got at least temperature
        if not parsed.get("temperature"):
            return {"ok": False, "data": None}

        data = {
            "temperature": parsed.get("temperature"),
            "pH": parsed.get("pH"),
            "water_level": parsed.get("water_level"),
            "conductivity": None,
            "turbidity": None,
            "color": None,
        }
        return {"ok": True, "data": data}
    except Exception:
        return {"ok": False, "data": None}


def read_arduino_f2(ser):
    """
    For F2: use temperature, water_level.
    Pressure is None for now; pH is optional.

    Returns:
      {
        "ok": True/False,
        "data": {
            "temperature": float or None,
            "water_level": float or None,
            "pressure": None,
            "pH": float or None,
        } or None
      }

    ok=False means: no valid line read yet.
    """
    if ser is None or not ser.is_open:
        return {"ok": False, "data": None}

    try:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            return {"ok": False, "data": None}

        parsed = parse_arduino_line(line)

        if not parsed.get("temperature"):
            return {"ok": False, "data": None}

        data = {
            "temperature": parsed.get("temperature"),
            "water_level": parsed.get("water_level"),
            "pressure": None,
            "pH": parsed.get("pH"),
        }
        return {"ok": True, "data": data}
    except Exception:
        return {"ok": False, "data": None}