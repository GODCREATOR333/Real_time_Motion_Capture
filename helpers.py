import numpy as np
from scipy import linalg, optimize, signal
import cv2 as cv
from scipy.spatial.transform import Rotation
import copy
import json
import os
import time
import numpy as np
import cv2 as cv
from KF import KalmanFilter
from pseyepy import Camera
from Singleton import Singleton


@Singleton
class Cameras:
    def __init__(self):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, "camera-params.json")
        f = open(filename)
        self.camera_params = json.load(f)

        self.cameras = Camera(fps=90, resolution=Camera.RES_SMALL, gain=10, exposure=100)
        self.num_cameras = len(self.cameras.exposure)
        print(self.num_cameras)

        self.is_capturing_points = False

        self.is_triangulating_points = False
        self.camera_poses = None

        self.is_locating_objects = False

        self.to_world_coords_matrix = None

        self.drone_armed = []

        self.num_objects = None

        self.kalman_filter = None

        self.socketio = None
        self.ser = None

        self.serialLock = None

        global cameras_init
        cameras_init = True

    def set_socketio(self, socketio):
        self.socketio = socketio
    
    def set_ser(self, ser):
        self.ser = ser

    def set_serialLock(self, serialLock):
        self.serialLock = serialLock

    def set_num_objects(self, num_objects):
        self.num_objects = num_objects
        self.drone_armed = [False for i in range(0, self.num_objects)]
    
    def edit_settings(self, exposure, gain):
        self.cameras.exposure = [exposure] * self.num_cameras
        self.cameras.gain = [gain] * self.num_cameras

    