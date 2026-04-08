import os
import cv2
import numpy as np
from collections import deque
from sklearn.metrics.pairwise import cosine_similarity
from pose.pose_embedder import PoseEmbedder
from pose.pose_detector import PoseDetector


class PoseMatcher:

    def __init__(self, db_path="hangitup", buffer_size=10):

        self.db_path = db_path
        self.embedder = PoseEmbedder()
        self.detector = PoseDetector()

        self.embeddings = {}
        self._buffer = deque(maxlen=buffer_size)
        self._build_database()

    def _build_database(self):

        print("Building pose database...")

        for file in os.listdir(self.db_path):

            path = os.path.join(self.db_path, file)

            img = cv2.imread(path)
            if img is None:
                continue

            landmarks = self.detector.detect(img)

            if landmarks:
                vec = self.embedder.embed(landmarks)
                self.embeddings[path] = vec
                print("Loaded:", file)

        print("Database size:", len(self.embeddings))

    def match(self, frame):

        landmarks = self.detector.detect(frame)

        if landmarks is None:
            return None

        self._buffer.append(self.embedder.embed(landmarks))
        query = np.mean(self._buffer, axis=0).reshape(1, -1)

        paths = list(self.embeddings.keys())
        vectors = np.array([self.embeddings[p] for p in paths])

        sims = cosine_similarity(query, vectors)[0]

        idx = np.argmax(sims)

        return paths[idx]