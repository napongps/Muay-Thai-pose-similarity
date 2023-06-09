import numpy as np
from mediapipe.framework.formats import landmark_pb2


def landmark_formatter(pose_landmarker_result):
    pose_landmarks_list = pose_landmarker_result.pose_world_landmarks
    people = len(pose_landmarks_list)

    all_landmarks = [[] for i in range(people)]

    for person in range(people):  # 2 people, len = 2
        person_landmarks = pose_landmarks_list[person]

        for i in range(33):
            all_landmarks[person].append([person_landmarks[i].x,
                                          person_landmarks[i].y,
                                          person_landmarks[i].z])

    return np.array(all_landmarks)
