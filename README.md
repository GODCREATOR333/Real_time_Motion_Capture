Real-Time Multi Camera Motion Capture and 3D Localization using cheap high speed (187 fps) PS3 eye cameras

# High-Speed Motion Capture System

Technical build and installation guide for a multi-camera PS3 Eye motion capture environment on Ubuntu with 3D Structure from Motion capabilities.

## Overview

This system enables high-speed (187 FPS) synchronized capture using PS3 Eye cameras with custom OpenCV builds supporting 3D reconstruction and trajectory planning.

## Camera Calibration
![Camera Calibration](/images/Screenshot%20from%202025-12-24%2016-32-40.png)

## Multi-Camera Setup
![PS3 Eye Cameras](/images/ps3_multi_camera.jpeg)


## Prerequisites

- Ubuntu 20.04 or later
- Python 3.8+
- 4+ GB RAM
- USB 2.0/3.0 ports for camera connections

## Installation

### 1. System Dependencies

Install required libraries for USB access and build tools:
```bash
sudo apt-get update
sudo apt-get install -y libusb-1.0-0-dev build-essential cmake python3-dev python3-pip
```

### 2. PS3 Eye Drivers (pseyepy)

Install pseyepy from source to enable 187 FPS capture:
```bash
git clone https://github.com/bensondaled/pseyepy.git
cd pseyepy
pip install .
```

**USB Permissions:** Configure udev rules for non-root camera access:
```bash
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="1415", ATTR{idProduct}=="2000", MODE="0666"' | sudo tee /etc/udev/rules.d/99-ps3eye.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### 3. Ceres Solver

Install Ceres Solver for Bundle Adjustment optimization:
```bash
# Install dependencies
sudo apt-get install -y libgoogle-glog-dev libgflags-dev libatlas-base-dev libsuitesparse-dev

# Build from source
git clone https://github.com/ceres-solver/ceres-solver.git
cd ceres-solver && mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

### 4. OpenCV with SFM Module

Build OpenCV from source with contrib modules:
```bash
# Clone repositories
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

# Configure build
cd opencv && mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D WITH_CERES=ON \
      -D BUILD_opencv_sfm=ON \
      -D BUILD_opencv_python3=ON ..

# Compile (takes 30-60 minutes)
make -j$(nproc)
sudo make install
sudo ldconfig
```

### 5. Trajectory Planning

Install Ruckig for jerk-limited motion profiles:
```bash
pip install ruckig
```

## Verification

Verify the installation:
```python
import cv2
print(f"OpenCV Version: {cv2.__version__}")
print(f"SFM Module Loaded: {hasattr(cv2, 'sfm')}")
```

Expected output:
```
OpenCV Version: 4.x.x
SFM Module Loaded: True
```

## Troubleshooting

### Camera Not Detected
- Check USB connection and try a different port
- Verify udev rules: `cat /etc/udev/rules.d/99-ps3eye.rules`
- Reconnect camera after rule changes

### OpenCV Build Fails
- Ensure sufficient disk space (10+ GB)
- Check Ceres installation: `ldconfig -p | grep ceres`
- Verify contrib path in cmake command

### SFM Module Missing
- Confirm `WITH_CERES=ON` and `BUILD_opencv_sfm=ON` in cmake
- Rebuild OpenCV after installing Ceres

## License

MIT

## Acknowledgments

- [pseyepy](https://github.com/bensondaled/pseyepy) - PS3 Eye camera interface
- [OpenCV](https://opencv.org/) - Computer vision library
- [Ceres Solver](http://ceres-solver.org/) - Non-linear optimization