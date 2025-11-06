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

    def _camera_read(self):
        frames, _ = self.cameras.read()

        for i in range(0, self.num_cameras):
            frames[i] = np.rot90(frames[i], k=self.camera_params[i]["rotation"])
            frames[i] = make_square(frames[i])
            frames[i] = cv.undistort(frames[i], self.get_camera_params(i)["intrinsic_matrix"], self.get_camera_params(i)["distortion_coef"])
            frames[i] = cv.GaussianBlur(frames[i],(9,9),0)
            kernel = np.array([[-2,-1,-1,-1,-2],
                               [-1,1,3,1,-1],
                               [-1,3,4,3,-1],
                               [-1,1,3,1,-1],
                               [-2,-1,-1,-1,-2]])
            frames[i] = cv.filter2D(frames[i], -1, kernel)
            frames[i] = cv.cvtColor(frames[i], cv.COLOR_RGB2BGR)

        if (self.is_capturing_points):
            image_points = []
            for i in range(0, self.num_cameras):
                frames[i], single_camera_image_points = self._find_dot(frames[i])
                image_points.append(single_camera_image_points)
            
            if (any(np.all(point[0] != [None,None]) for point in image_points)):
                if self.is_capturing_points and not self.is_triangulating_points:
                    self.socketio.emit("image-points", [x[0] for x in image_points])
                elif self.is_triangulating_points:
                    errors, object_points, frames = find_point_correspondance_and_object_points(image_points, self.camera_poses, frames)

                    # convert to world coordinates
                    for i, object_point in enumerate(object_points):
                        new_object_point = np.array([[-1,0,0],[0,-1,0],[0,0,1]]) @ object_point
                        new_object_point = np.concatenate((new_object_point, [1]))
                        new_object_point = np.array(self.to_world_coords_matrix) @ new_object_point
                        new_object_point = new_object_point[:3] / new_object_point[3]
                        new_object_point[1], new_object_point[2] = new_object_point[2], new_object_point[1]
                        object_points[i] = new_object_point

                    objects = []
                    filtered_objects = []
                    if self.is_locating_objects:
                        objects = locate_objects(object_points, errors)
                        filtered_objects = self.kalman_filter.predict_location(objects)
                        
                        if len(filtered_objects) != 0:
                            for filtered_object in filtered_objects:
                                if self.drone_armed[filtered_object['droneIndex']]:
                                    filtered_object["heading"] = round(filtered_object["heading"], 4)

                                    serial_data = { 
                                        "pos": [round(x, 4) for x in filtered_object["pos"].tolist()] + [filtered_object["heading"]],
                                        "vel": [round(x, 4) for x in filtered_object["vel"].tolist()]
                                    }
                                    with self.serialLock:
                                        self.ser.write(f"{filtered_object['droneIndex']}{json.dumps(serial_data)}".encode('utf-8'))
                                        time.sleep(0.001)
                            
                        for filtered_object in filtered_objects:
                            filtered_object["vel"] = filtered_object["vel"].tolist()
                            filtered_object["pos"] = filtered_object["pos"].tolist()
                    
                    self.socketio.emit("object-points", {
                        "object_points": object_points.tolist(), 
                        "errors": errors.tolist(), 
                        "objects": [{k:(v.tolist() if isinstance(v, np.ndarray) else v) for (k,v) in object.items()} for object in objects], 
                        "filtered_objects": filtered_objects
                    })
        
        return frames

    def get_frames(self):
        frames = self._camera_read()
        #frames = [add_white_border(frame, 5) for frame in frames]

        return np.hstack(frames)

    def _find_dot(self, img):
        # img = cv.GaussianBlur(img,(5,5),0)
        grey = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        grey = cv.threshold(grey, 255*0.2, 255, cv.THRESH_BINARY)[1]
        contours,_ = cv.findContours(grey, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        img = cv.drawContours(img, contours, -1, (0,255,0), 1)

        image_points = []
        for contour in contours:
            moments = cv.moments(contour)
            if moments["m00"] != 0:
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])
                cv.putText(img, f'({center_x}, {center_y})', (center_x,center_y - 15), cv.FONT_HERSHEY_SIMPLEX, 0.3, (100,255,100), 1)
                cv.circle(img, (center_x,center_y), 1, (100,255,100), -1)
                image_points.append([center_x, center_y])

        if len(image_points) == 0:
            image_points = [[None, None]]

        return img, image_points

    def start_capturing_points(self):
        self.is_capturing_points = True

    def stop_capturing_points(self):
        self.is_capturing_points = False

    def start_trangulating_points(self, camera_poses):
        self.is_capturing_points = True
        self.is_triangulating_points = True
        self.camera_poses = camera_poses
        self.kalman_filter = KalmanFilter(self.num_objects)

    def stop_trangulating_points(self):
        self.is_capturing_points = False
        self.is_triangulating_points = False
        self.camera_poses = None

    def start_locating_objects(self):
        self.is_locating_objects = True

    def stop_locating_objects(self):
        self.is_locating_objects = False
    
    def get_camera_params(self, camera_num):
        return {
            "intrinsic_matrix": np.array(self.camera_params[camera_num]["intrinsic_matrix"]),
            "distortion_coef": np.array(self.camera_params[camera_num]["distortion_coef"]),
            "rotation": self.camera_params[camera_num]["rotation"]
        }
    
    def set_camera_params(self, camera_num, intrinsic_matrix=None, distortion_coef=None):
        if intrinsic_matrix is not None:
            self.camera_params[camera_num]["intrinsic_matrix"] = intrinsic_matrix
        
        if distortion_coef is not None:
            self.camera_params[camera_num]["distortion_coef"] = distortion_coef


def calculate_reprojection_errors(image_points, object_points, camera_poses):
    errors = np.array([])
    for image_points_i, object_point in zip(image_points, object_points):
        error = calculate_reprojection_error(image_points_i, object_point, camera_poses)
        if error is None:
            continue
        errors = np.concatenate([errors, [error]])

    return errors


def calculate_reprojection_error(image_points, object_point, camera_poses):
    cameras = Cameras.instance()

    image_points = np.array(image_points)
    none_indicies = np.where(np.all(image_points == None, axis=1))[0]
    image_points = np.delete(image_points, none_indicies, axis=0)
    camera_poses = np.delete(camera_poses, none_indicies, axis=0)

    if len(image_points) <= 1:
        return None

    image_points_t = image_points.transpose((0,1))

    errors = np.array([])
    for i, camera_pose in enumerate(camera_poses):
        if np.all(image_points[i] == None, axis=0):
            continue
        projected_img_points, _ = cv.projectPoints(
            np.expand_dims(object_point, axis=0).astype(np.float32), 
            np.array(camera_pose["R"], dtype=np.float64), 
            np.array(camera_pose["t"], dtype=np.float64), 
            cameras.get_camera_params(i)["intrinsic_matrix"], 
            np.array([])
        )
        projected_img_point = projected_img_points[:,0,:][0]
        errors = np.concatenate([errors, (image_points_t[i]-projected_img_point).flatten() ** 2])
    
    return errors.mean()


def bundle_adjustment(image_points, camera_poses, socketio):
    cameras = Cameras.instance()

    def params_to_camera_poses(params):
        focal_distances = []
        num_cameras = int((params.size-1)/7)+1
        camera_poses = [{
            "R": np.eye(3),
            "t": np.array([0,0,0], dtype=np.float32)
        }]
        focal_distances.append(params[0])
        for i in range(0, num_cameras-1):
            focal_distances.append(params[i*7+1])
            camera_poses.append({
                "R": Rotation.as_matrix(Rotation.from_rotvec(params[i*7 + 2 : i*7 + 3 + 2])),
                "t": params[i*7 + 3 + 2 : i*7 + 6 + 2]
            })

        return camera_poses, focal_distances

    def residual_function(params):
        camera_poses, focal_distances = params_to_camera_poses(params)
        for i in range(0, len(camera_poses)):
            intrinsic = cameras.get_camera_params(i)["intrinsic_matrix"]
            intrinsic[0, 0] = focal_distances[i]
            intrinsic[1, 1] = focal_distances[i]
            # cameras.set_camera_params(i, intrinsic)
        object_points = triangulate_points(image_points, camera_poses)
        errors = calculate_reprojection_errors(image_points, object_points, camera_poses)
        errors = errors.astype(np.float32)
        socketio.emit("camera-pose", {"camera_poses": camera_pose_to_serializable(camera_poses)})
        
        return errors

    focal_distance = cameras.get_camera_params(0)["intrinsic_matrix"][0,0]
    init_params = np.array([focal_distance])
    for i, camera_pose in enumerate(camera_poses[1:]):
        rot_vec = Rotation.as_rotvec(Rotation.from_matrix(camera_pose["R"])).flatten()
        focal_distance = cameras.get_camera_params(i)["intrinsic_matrix"][0,0]
        init_params = np.concatenate([init_params, [focal_distance]])
        init_params = np.concatenate([init_params, rot_vec])
        init_params = np.concatenate([init_params, camera_pose["t"].flatten()])

    res = optimize.least_squares(
        residual_function, init_params, verbose=2, loss="cauchy", ftol=1E-2
    )
    return params_to_camera_poses(res.x)[0]



