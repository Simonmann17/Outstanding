import cv2
import os
import numpy as np


class FaceMatcher:

    def __init__(self, db_path="hangitup"):
        self.db_path = db_path
        self.match_interval = 30
        self.frame_count = 0
        self.current_match = None

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.label_to_path = {}

        self._train()

    def _train(self):
        faces, labels = [], []

        for idx, file in enumerate(os.listdir(self.db_path)):
            path = os.path.join(self.db_path, file)
            img = cv2.imread(path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            if len(detected) > 0:
                x, y, w, h = detected[0]
                face = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
                faces.append(face)
                labels.append(idx)
                self.label_to_path[idx] = path

        if faces:
            self.recognizer.train(faces, np.array(labels))
            print(f"Trained on {len(faces)} player images")
        else:
            print("Warning: no faces detected in player images during training")

    def update(self, frame):
        self.frame_count += 1
        if self.frame_count % self.match_interval != 0:
            return self.current_match

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(detected) > 0:
            x, y, w, h = detected[0]
            face = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
            try:
                label, _ = self.recognizer.predict(face)
                self.current_match = self.label_to_path.get(label)
            except Exception:
                pass

        return self.current_match
