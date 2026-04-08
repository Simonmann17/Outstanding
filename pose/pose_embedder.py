import numpy as np

# (landmark_index, weight)
# Higher weight = more influence on cosine similarity match
LANDMARKS = [
    # Face
    (0,  2.0),  # nose
    (9,  2.0),  # mouth left
    (10, 2.0),  # mouth right
    # Shoulders
    (11, 1.0),  # left shoulder
    (12, 1.0),  # right shoulder
    # Elbows
    (13, 1.5),  # left elbow
    (14, 1.5),  # right elbow
    # Wrists
    (15, 2.0),  # left wrist
    (16, 2.0),  # right wrist
    # Fingers
    (17, 2.0),  # left pinky
    (18, 2.0),  # right pinky
    (19, 2.0),  # left index
    (20, 2.0),  # right index
    (21, 2.0),  # left thumb
    (22, 2.0),  # right thumb
    # Hips (additional reference, low weight)
    (23, 0.5),  # left hip
    (24, 0.5),  # right hip
]


class PoseEmbedder:

    def embed(self, landmarks):

        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

        # Anchor: shoulder midpoint (reliable, almost always visible)
        shoulder_mid = (coords[11] + coords[12]) / 2

        # Scale: shoulder-to-hip torso length
        hip_mid = (coords[23] + coords[24]) / 2
        torso = np.linalg.norm(shoulder_mid - hip_mid)

        vec = []
        for idx, weight in LANDMARKS:
            point = (coords[idx] - shoulder_mid)
            if torso > 0:
                point /= torso
            vec.append(point * weight)

        return np.array(vec).flatten()
