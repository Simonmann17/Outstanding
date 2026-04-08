import cv2
import os


class ImageDisplay:

    def __init__(self, db_path="hangitup"):

        self.images = {}

        for file in os.listdir(db_path):

            path = os.path.join(db_path, file)

            img = cv2.imread(path)

            if img is not None:
                self.images[path] = img

        cv2.namedWindow("Camera")
        cv2.namedWindow("NBA Match")

    def show(self, frame, match):

        cv2.imshow("Camera", frame)

        if match and match in self.images:
            cv2.imshow("NBA Match", self.images[match])