import cv2
import os


class ImageDisplay:

    def __init__(self):
        self.db_path = "hangitup"
        self.images = {}

        for file in os.listdir(self.db_path):
            path = os.path.join(self.db_path, file)
            img = cv2.imread(path)
            if img is not None:
                self.images[path] = img

    def show(self, frame, match_path):
        cv2.imshow("Camera Feed", frame)

        if match_path and match_path in self.images:
            cv2.imshow("Your NBA Twin", self.images[match_path])
