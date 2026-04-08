import cv2
import pygame

from camera.camera import Camera
from matcher.pose_matcher import PoseMatcher
from display.image_display import ImageDisplay


def main():

    camera = Camera()

	# initialize audio
    pygame.mixer.init()
    pygame.mixer.music.load("audio/music.mp3")
    pygame.mixer.music.play(-1)  # loop forever

    matcher = PoseMatcher("hangitup")

    display = ImageDisplay("hangitup")

    while True:

        frame = camera.get_frame()

        if frame is None:
            print("Frame failed")
            break

        match = matcher.match(frame)

        display.show(frame, match)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    camera.release()
    
    pygame.mixer.music.stop()
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()