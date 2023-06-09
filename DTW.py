import numpy as np
from local_joint import extract_local_angle, distance_local_angle
from cosine_distance import cosine_distance
from fastdtw import fastdtw


def dist_local_joint(pose_landmarker_result_model, pose_landmarker_result_input):
    vector_model = extract_local_angle(pose_landmarker_result_model)
    vector_input = extract_local_angle(pose_landmarker_result_input)

    return distance_local_angle(vector_model, vector_input)


def dist_cosine_distance(pose_landmarker_result_model, pose_landmarker_result_input):
    vector_model = find_limb(pose_landmarker_result_model)
    vector_input = find_limb(pose_landmarker_result_input)

    return cosine_distance(vector_model, vector_input)

def fast_dtw(pose_landmarker_result_model, pose_landmarker_result_input, distance_function):
    return fastdtw(pose_landmarker_result_model, pose_landmarker_result_input, dist=distance_function)
