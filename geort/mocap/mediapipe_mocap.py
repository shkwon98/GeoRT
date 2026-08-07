# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from geort.utils.path import get_hand_landmarker_path


class MediaPipeHandProcessor:
    def __init__(self):
        # we will need to track the orientation, and do some low pass filtering.
        self.last_rotation = None

    def ema_rotation_matrix(self, R1, R2, alpha):
        from scipy.spatial.transform import Rotation as R
        # Convert rotation matrices to quaternions
        q1 = R.from_matrix(R1).as_quat()
        q2 = R.from_matrix(R2).as_quat()

        # Create a slerp object and interpolate
        interpolated = self.slerp(q1, q2, alpha)

        # Convert back to a rotation matrix
        R_interpolated = R.from_quat(interpolated).as_matrix()
        return R_interpolated

    def slerp(self, q1, q2, t):
        q1 = np.array(q1)
        q2 = np.array(q2)
        dot_product = np.dot(q1, q2)
        if dot_product < 0.0:
            q1 = -q1
            dot_product = -dot_product

        if dot_product > 0.9995:  # Quaternions are close, use linear interpolation
            result = (1.0 - t) * q1 + t * q2
            return result / np.linalg.norm(result)

        theta_0 = np.arccos(dot_product)  # Angle between quaternions
        sin_theta_0 = np.sin(theta_0)      # Sine of the angle
        theta = theta_0 * t                # Angle for the interpolation
        # Sine of the angle for the interpolation
        sin_theta = np.sin(theta)
        sin_theta_1 = np.sin(theta_0 - theta)  # Sine of the remaining angle

        q_interpolated = (sin_theta_1 * q1 + sin_theta * q2) / sin_theta_0
        return q_interpolated / np.linalg.norm(q_interpolated)

    def forward(self, hand_detection_result, apply_ema=False):
        # fine, apply_ema seems futile for MediaPipe.
        z_axis = hand_detection_result[9] - hand_detection_result[0]
        z_axis = z_axis / np.linalg.norm(z_axis)
        y_axis_aux = hand_detection_result[5] - hand_detection_result[13]
        y_axis_aux = y_axis_aux / np.linalg.norm(y_axis_aux)

        x_axis = np.cross(y_axis_aux, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        tranlation_base = hand_detection_result[0]
        rotation_base = np.array([x_axis, y_axis, z_axis]).transpose()

        if apply_ema:
            if self.last_rotation is not None:
                rotation_base = self.ema_rotation_matrix(
                    self.last_rotation, rotation_base, 0.75)
            self.last_rotation = rotation_base

        transform = np.eye(4)
        transform[:3, :3] = rotation_base
        transform[:3, 3] = tranlation_base

        transform_inv = np.linalg.inv(transform)
        hand_detection_result_np = np.array(hand_detection_result)
        hand_detection_result_np = np.concatenate(
            (np.array(hand_detection_result_np), np.ones((21, 1))), axis=-1)

        hand_detection_result_np = hand_detection_result_np @ transform_inv.transpose()

        return hand_detection_result_np


class MediaPipeHandDetector:
    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

    def __init__(self):
        import cv2
        import mediapipe as mp
        from mediapipe import solutions
        from mediapipe.framework.formats import landmark_pb2
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self.cv2 = cv2
        self.mp = mp
        self.solutions = solutions
        self.landmark_pb2 = landmark_pb2
        base_options = python.BaseOptions(
            model_asset_path=str(get_hand_landmarker_path()))
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                               num_hands=2)
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.processor = MediaPipeHandProcessor()

        return

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = self.landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                self.landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            self.solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                self.solutions.hands.HAND_CONNECTIONS,
                self.solutions.drawing_styles.get_default_hand_landmarks_style(),
                self.solutions.drawing_styles.get_default_hand_connections_style())

            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - \
                MediaPipeHandDetector.MARGIN

            # Draw handedness (left or right hand) on the image.
            self.cv2.putText(annotated_image, f"{handedness[0].category_name}",
                        (text_x, text_y), self.cv2.FONT_HERSHEY_DUPLEX,
                        MediaPipeHandDetector.FONT_SIZE,
                        MediaPipeHandDetector.HANDEDNESS_TEXT_COLOR,
                        MediaPipeHandDetector.FONT_THICKNESS,
                        self.cv2.LINE_AA)

        return annotated_image

    def numpy_to_mp_image(self, image_np):
        image_np = self.cv2.cvtColor(image_np, self.cv2.COLOR_RGB2BGR)
        # Create a MediaPipe image
        mp_image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB, data=image_np
        )
        return mp_image

    def detect(self, rgb_image, hand_side):
        detection_result = self.detector.detect(
            self.numpy_to_mp_image(rgb_image))

        detected = False
        coordinates = []
        world_coordinates = []
        canonical_coordinates = []
        selected = next(
            (
                index
                for index, handedness in enumerate(detection_result.handedness)
                if handedness[0].category_name.lower() == hand_side
            ),
            None,
        )

        if selected is not None:
            hand_landmarks = detection_result.hand_landmarks[selected]
            for landmark in hand_landmarks:
                coordinates.append([landmark.x, landmark.y, landmark.z])
            detected = True
            world_landmarks = detection_result.hand_world_landmarks[selected]
            for landmark in world_landmarks:
                world_coordinates.append([landmark.x, landmark.y, landmark.z])
        annotated_image = self.draw_landmarks_on_image(
            rgb_image, detection_result)

        if detected:
            canonical_coordinates = self.processor.forward(
                np.array(coordinates))[..., :3]

        # print(annotated_image.shape)
        return {
            'detected': detected,
            "annotated_img": annotated_image,
            "coordinates": np.array(coordinates),
            "canonical_coordinates": canonical_coordinates,
            "world_coordinates": np.array(world_coordinates)
        }


class MediaPipeMocap:
    def __init__(self, camera="realsense", device_index=0, hand_side="right"):
        import cv2

        if hand_side not in {"left", "right"}:
            raise ValueError("hand_side must be 'left' or 'right'")
        if camera == "realsense":
            from geort.mocap.camera.realsense import RealSenseCamera

            self.camera = RealSenseCamera()
        elif camera == "webcam":
            from geort.mocap.camera.webcam import WebcamCamera

            self.camera = WebcamCamera(device_index)
        else:
            raise ValueError("camera must be 'realsense' or 'webcam'")
        self.cv2 = cv2
        self.detector = MediaPipeHandDetector()
        self.hand_side = hand_side
        self.status = 'idle'

    def get(self):
        # Run the mocap system.
        rgb = self.camera.get_frame()["rgb"]
        if rgb is None:
            return {"status": self.status, "result": None}
        result = self.detector.detect(rgb, self.hand_side)

        # Show the live detection.
        self.cv2.imshow("detection", result["annotated_img"])

        # Keyboard Control.
        key = self.cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            self.status = 'quit'
        elif key == ord('s'):
            self.status = 'recording'
        elif key == ord('e'):
            self.status = 'idle'

        detection = None
        if result["detected"] and len(result["world_coordinates"]) == 21:
            detection = result["world_coordinates"]
        return {'status': self.status, "result": detection}

    def close(self):
        self.camera.release()
        self.cv2.destroyAllWindows()
