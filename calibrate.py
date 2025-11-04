import numpy as np
import cv2 as cv
import json

# --- SETTINGS ---
# Change this to match your checkerboard internal corners (Width x Height)
# Example dynamic selection
BOARD_CHOICES = {
    "small": (8,6),   # 9x7 squares
    "medium": (10,7), # 11x8 squares
    "large": (13,9)   # 14x10 squares
}

CHESSBOARD_SIZE = BOARD_CHOICES["medium"]  # change as needed
# Size of a square in millimeters (optional, for scale)
SQUARE_SIZE = 25 

# Termination criteria for sub-pixel accuracy
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE

# Arrays to store object points and image points from all images
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane

# Initialize Camera (0 is usually the first PS3 Eye)
cap = cv.VideoCapture(0)

print("--- PS3 Eye Calibrator ---")
print("1. Hold the checkerboard in front of the camera.")
print("2. When corners are detected (colorful lines), press 'S' to save the frame.")
print("3. Capture at least 15-20 frames from different angles/distances.")
print("4. Press 'Q' when finished to calculate and save.")

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if ret == True:
        # Draw the corners to show the user
        vis_frame = frame.copy()
        cv.drawChessboardCorners(vis_frame, CHESSBOARD_SIZE, corners, ret)
        cv.imshow('Calibration', vis_frame)
        
        key = cv.waitKey(1) & 0xFF
        if key == ord('s'):
            objpoints.append(objp)
            # Refine corners for better accuracy
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            count += 1
            print(f"Stored frame {count}")
    else:
        cv.imshow('Calibration', frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

if len(objpoints) > 10:
    print("Calculating calibration... please wait.")
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Prepare data for JSON
    # We convert numpy arrays to lists so they can be saved in JSON
    calib_data = {
        "camera_matrix": mtx.tolist(),
        "distortion_coefficients": dist.tolist(),
        "resolution": [frame.shape[1], frame.shape[0]],
        "reprojection_error": float(ret)
    }

    with open('camera-params.json', 'w') as f:
        json.dump(calib_data, f, indent=4)

    print("\nCalibration Complete!")
    print(f"Reprojection Error: {ret:.4f} (Lower is better, < 0.5 is great)")
    print("Saved to camera-params.json")
else:
    print("Not enough frames captured. Calibration failed.")