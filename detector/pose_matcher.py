import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import urllib.request
from sklearn.metrics.pairwise import cosine_similarity

UPPER_BODY = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
MODEL_PATH = "pose_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"


class PoseMatcher:

    def __init__(self, db_path="hangitup"):
        self.db_path = db_path
        self.match_interval = 5
        self.frame_count = 0
        self.current_match = None

        if not os.path.exists(MODEL_PATH):
            print("Downloading pose model...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

        video_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(video_options)

        self.embeddings = {}
        self._train()

    def _normalize(self, landmarks):
        all_coords = np.array([[lm.x, lm.y] for lm in landmarks])

        hip_mid = (all_coords[23] + all_coords[24]) / 2
        shoulder_mid = (all_coords[11] + all_coords[12]) / 2
        torso_size = np.linalg.norm(shoulder_mid - hip_mid)

        coords = all_coords[UPPER_BODY]
        coords -= hip_mid
        if torso_size > 0:
            coords /= torso_size

        return coords.flatten()

    def _train(self):
        static_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
        )
        static_landmarker = vision.PoseLandmarker.create_from_options(static_options)

        for file in os.listdir(self.db_path):
            path = os.path.join(self.db_path, file)
            img = cv2.imread(path)
            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = static_landmarker.detect(mp_image)

            if result.pose_landmarks:
                self.embeddings[path] = self._normalize(result.pose_landmarks[0])
                print(f"Pose loaded: {file}")
            else:
                print(f"No pose detected: {file}")

        static_landmarker.close()
        print(f"Ready with {len(self.embeddings)} player poses")

    def update(self, frame):
        self.frame_count += 1
        if self.frame_count % self.match_interval != 0:
            return self.current_match

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = self.frame_count * 33  # ~30fps
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks and self.embeddings:
            query = self._normalize(result.pose_landmarks[0]).reshape(1, -1)
            paths = list(self.embeddings.keys())
            vectors = np.array([self.embeddings[p] for p in paths])
            sims = cosine_similarity(query, vectors)[0]
            self.current_match = paths[np.argmax(sims)]

        return self.current_match
