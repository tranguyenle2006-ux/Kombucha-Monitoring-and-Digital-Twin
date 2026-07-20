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
        time.sleep(2)
        return ser
    except Exception:
        return None

def parse_arduino_line(line):
    """
    Expected format from your Arduino:
      Temperature: 26.80 C, Water Level: 74 %, pH: 4.21

    We will only use:
      - Temperature
      - pH (for F1)
    Water level is ignored in the app now.

    Returns:
      {
        "temperature": float or None,
        "pH": float or None
      }
    """
    data = {}

    t_match = re.search(r"Temperature:\s*([-+]?\d*\.?\d+)", line)
    if t_match:
        data["temperature"] = float(t_match.group(1))

    # pH is only used in F1; still parsed here
    ph_match = re.search(r"pH:\s*([-+]?\d*\.?\d+)", line)
    if ph_match:
        data["pH"] = float(ph_match.group(1))

    return data


def read_arduino_once(ser):
    """
    Try to read one valid line from Arduino.
    Returns:
      {
        "ok": True/False,
        "temperature": float or None,
        "pH": float or None
      }
    """
    if ser is None or not ser.is_open:
        return {"ok": False, "temperature": None, "pH": None}

    try:
        # Try a few times to get a complete line
        for _ in range(5):
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            parsed = parse_arduino_line(line)
            if parsed.get("temperature") is not None:
                return {
                    "ok": True,
                    "temperature": parsed["temperature"],
                    "pH": parsed.get("pH")
                }
        return {"ok": False, "temperature": None, "pH": None}
    except Exception:
        return {"ok": False, "temperature": None, "pH": None}