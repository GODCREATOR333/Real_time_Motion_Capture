import cv2 as cv
import numpy as np
from LowPassFilter import LowPassFilter
import time

class KalmanFilter:
    def __init__(self, num_objects):
        state_dim = 9
        measurement_dim = 6
        dt = 0.1
        self.kalmans = []
        self.prev_measurement_time = 0
        self.prev_positions = []

        self.low_pass_filter_xy = []
        self.low_pass_filter_z = []
        self.heading_low_pass_filter = []
        self.num_objects = num_objects

        for i in range(num_objects):
            self.prev_positions.append([0,0,0])
            self.kalmans.append(cv.KalmanFilter(state_dim, measurement_dim))
            self.kalmans[i].transitionMatrix = np.array([[1, 0, 0, dt, 0, 0, 0.5*dt**2, 0, 0],
                                                    [0, 1, 0, 0, dt, 0, 0, 0.5*dt**2, 0],
                                                    [0, 0, 1, 0, 0, dt, 0, 0, 0.5*dt**2],
                                                    [0, 0, 0, 1, 0, 0, dt, 0, 0],
                                                    [0, 0, 0, 0, 1, 0, 0, dt, 0],
                                                    [0, 0, 0, 0, 0, 1, 0, 0, dt],
                                                    [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0, 1, 0],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)

            self.kalmans[i].processNoiseCov = np.eye(state_dim, dtype=np.float32) * 1e-2
            self.kalmans[i].measurementNoiseCov = np.eye(measurement_dim, dtype=np.float32) * 1e0
            self.kalmans[i].measurementMatrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 1, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 1, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 1, 0, 0, 0]], dtype=np.float32)
            
            self.kalmans[i].statePost = np.zeros((9,1), dtype=np.float32)


            self.low_pass_filter_xy.append(LowPassFilter(cutoff_frequency=20, sampling_frequency=60.0, dims=2))
            self.low_pass_filter_z.append(LowPassFilter(cutoff_frequency=20, sampling_frequency=60.0, dims=1))
            self.heading_low_pass_filter.append(LowPassFilter(cutoff_frequency=20, sampling_frequency=60.0, dims=1))
    