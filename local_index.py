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
from utils import camera_pose_to_serializable, calculate_reprojection_errors, bundle_adjustment, Cameras, triangulate_points
from kalman_filter import KalmanFilter





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



# --- HELPER FUNCTIONS (Transplanted from your API routes) ---

def run_calibration(captured_points):
    """Calculates camera poses from captured points"""
    print("Calculating camera pose... please wait.")
    image_points = np.array(captured_points)
    image_points_t = image_points.transpose((1, 0, 2))

    camera_poses = [{"R": np.eye(3), "t": np.array([[0],[0],[0]], dtype=np.float32)}]
    
    for camera_i in range(0, cameras.num_cameras - 1):
        c1_pts = image_points_t[camera_i]
        c2_pts = image_points_t[camera_i+1]
        
        mask = np.where(np.all(c1_pts != None, axis=1) & np.all(c2_pts != None, axis=1))[0]
        c1_pts = np.take(c1_pts, mask, axis=0).astype(np.float32)
        c2_pts = np.take(c2_pts, mask, axis=0).astype(np.float32)

        F, _ = cv.findFundamentalMat(c1_pts, c2_pts, cv.FM_RANSAC, 1, 0.99999)
        E = cv.sfm.essentialFromFundamental(F, cameras.get_camera_params(0)["intrinsic_matrix"], cameras.get_camera_params(1)["intrinsic_matrix"])
        possible_Rs, possible_ts = cv.sfm.motionFromEssential(E)

        R, t, max_points = None, None, 0
        for i in range(4):
            obj_pts = triangulate_points(np.hstack([np.expand_dims(c1_pts, 1), np.expand_dims(c2_pts, 1)]), 
                                        np.concatenate([[camera_poses[-1]], [{"R": possible_Rs[i], "t": possible_ts[i]}]]))
            cam_frame = np.array([possible_Rs[i].T @ p for p in obj_pts])
            valid = np.sum(obj_pts[:,2] > 0) + np.sum(cam_frame[:,2] > 0)
            if valid > max_points:
                max_points, R, t = valid, possible_Rs[i], possible_ts[i]

        camera_poses.append({"R": R @ camera_poses[-1]["R"], "t": camera_poses[-1]["t"] + (camera_poses[-1]["R"] @ t)})

    refined_poses = bundle_adjustment(image_points, camera_poses, socketio)
    print("Calibration Complete.")
    return refined_poses


def display_frames_grid(frames, window_name="Real-Time Motion tracking and localization"):
    """Display 4 camera feeds in a 2x2 grid"""
    if frames is None:
        return
    
    if not isinstance(frames, np.ndarray):
        return
    
    h, w = frames.shape[:2]
    
    # If already a single image (concatenated horizontally)
    # Split into 4 equal parts
    if w > h * 2:  # Wider than tall = horizontal concat
        frame_width = w // 4
        
        # Split the horizontal concat into 4 frames
        cam0 = frames[:, 0:frame_width]
        cam1 = frames[:, frame_width:frame_width*2]
        cam2 = frames[:, frame_width*2:frame_width*3]
        cam3 = frames[:, frame_width*3:frame_width*4]
        
        # Arrange in 2x2 grid
        top_row = np.hstack([cam0, cam1])
        bottom_row = np.hstack([cam2, cam3])
        grid = np.vstack([top_row, bottom_row])
        
        cv.imshow(window_name, grid)
    else:
        # If it's already in correct format or unknown, just show it
        cv.imshow(window_name, frames)



# --- MAIN LOOP ---

def main():
    global ser
    # Initialize Camera Singleton
    cameras.set_socketio(socketio)
    cameras.set_ser(ser)
    cameras.set_serialLock(serialLock)
    cameras.set_num_objects(NUM_OBJECTS)

    # Internal States
    is_capturing = False
    is_locating = False
    is_triangulating = False
    active_poses = None
    
    print("\n" + "="*30)
    print("LOCAL MOCAP CONTROL")
    print("="*30)
    print("C: Start/Stop Capture Points")
    print("P: Calculate Camera Poses (Run after 'C')")
    print("L: Start/Stop Locating Objects")
    print("T: Start/Stop Live Triangulation")
    print("A: Toggle Arm Drones")
    print("Q: Quit")
    print("="*30 + "\n")

    while True:
        # 1. Grab frames
        frames = cameras.get_frames()
        
        # 2. Display window
        if frames is not None:
            display_frames_grid(frames)

        # 3. Handle Key Inputs
        key = cv.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
            
        elif key == ord('c'): # Capture Points
            if not is_capturing:
                cameras.start_capturing_points()
                is_capturing = True
                print(">>> CAPTURE STARTED")
            else:
                cameras.stop_capturing_points()
                is_capturing = False
                print(">>> CAPTURE STOPPED")

        elif key == ord('p'): # Calculate Poses
            # Note: This uses points stored inside the cameras object
            if hasattr(cameras, 'captured_points') and len(cameras.captured_points) > 5:
                active_poses = run_calibration(cameras.captured_points)
                print("Pose saved to memory.")
            else:
                print("Error: Not enough points captured. Press 'C' first.")

        elif key == ord('l'): # Locate Objects
            if not is_locating:
                cameras.start_locating_objects()
                is_locating = True
                print(">>> LOCATING STARTED")
            else:
                cameras.stop_locating_objects()
                is_locating = False
                print(">>> LOCATING STOPPED")

        elif key == ord('t'): # Triangulate
            if active_poses is None:
                print("Error: Calibrate (P) before Triangulating.")
                continue
            if not is_triangulating:
                # Use default identity for world matrix if not set
                if not hasattr(cameras, 'to_world_coords_matrix'):
                    cameras.to_world_coords_matrix = np.eye(4)
                cameras.start_trangulating_points(active_poses)
                is_triangulating = True
                print(">>> LIVE MOCAP STARTED")
            else:
                cameras.stop_trangulating_points()
                is_triangulating = False
                print(">>> LIVE MOCAP STOPPED")

        elif key == ord('a'): # Arm Drones
            armed_state = [True] * NUM_OBJECTS
            cameras.drone_armed = armed_state
            if ser:
                for i in range(NUM_OBJECTS):
                    with serialLock:
                        ser.write(f"{i}{json.dumps({'armed': True})}".encode('utf-8'))
                print(">>> DRONES ARMED")
            else:
                print(">>> DRONES ARMED (Simulated)")

    if ser:
        ser.close()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
