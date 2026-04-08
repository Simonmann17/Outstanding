import cv2
import time


class Camera:

    def __init__(self, index: int = 0):
        self.cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)

        if not self.cap.isOpened() and index == 0:
            # fallback: try index 1 (external camera)
            self.cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")

        # warmup frames
        for _ in range(10):
            self.cap.read()
            time.sleep(0.03)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()