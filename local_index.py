import cv2 as cv
import numpy as np
import json
import time
import serial
import threading
import copy
from scipy import linalg
from ruckig import InputParameter, OutputParameter, Result, Ruckig

# Import the helper modules from the repo
from helpers import camera_pose_to_serializable, calculate_reprojection_errors, bundle_adjustment, Cameras, triangulate_points
from KF import KalmanFilter

# --- CONFIGURATION ---
SERIAL_PORT = "/dev/cu.usbserial-02X2K2GE" # Change this to your port (e.g., "COM3" on Windows)
BAUD_RATE = 1000000
NUM_OBJECTS = 2

# --- MOCK SOCKETIO CLASS ---
# This prevents the 'Cameras' class from breaking when it tries to talk to the web
class MockSocketIO:
    def emit(self, event, data):
        if event == "fps":
            return # Don't spam FPS to console
        print(f"[UI Update] {event}: {list(data.keys()) if isinstance(data, dict) else data}")

# --- GLOBAL STATE ---
serialLock = threading.Lock()
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, write_timeout=1)
    print(f"Connected to Serial: {SERIAL_PORT}")
except Exception as e:
    print(f"Serial Error: {e}. Running without hardware.")
    ser = None

socketio = MockSocketIO()
cameras = Cameras.instance()