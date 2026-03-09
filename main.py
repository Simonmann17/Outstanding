import cv2
import pygame

from detector.camera import Camera
from detector.face_mesh import FaceMatcher
from display.image_display import ImageDisplay


def main():

    # initialize audio
    pygame.mixer.init()
    pygame.mixer.music.load("audio/music.mp3")
    pygame.mixer.music.play(-1)  # loop forever

    camera = Camera()
    matcher = FaceMatcher(db_path="hangitup")
    display = ImageDisplay()

    while True:

        frame = camera.get_frame()
        if frame is None:
            break

        match_path = matcher.update(frame)
        display.show(frame, match_path)

        # press ESC to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    camera.release()
    pygame.mixer.music.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
