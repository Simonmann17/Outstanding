import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import os
import urllib.request

MODEL_PATH = "pose_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"


class PoseDetector:

    def __init__(self):

        if not os.path.exists(MODEL_PATH):
            print("Downloading pose model...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
        )

        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.frame_id = 0

    def detect(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        self.frame_id += 1
        timestamp = self.frame_id * 33

        result = self.detector.detect_for_video(mp_image, timestamp)

        if result.pose_landmarks:
            return result.pose_landmarks[0]

        return None