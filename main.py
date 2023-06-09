from Detector_video import Video_detector
from Detector_live_stream import Live_detector, print_result
from DTW import fast_dtw, dist_local_joint, dist_cosine_distance
from landmark_formatter import landmark_formatter
import numpy as np
import cv2
import matplotlib.pyplot as plt

detected_model = Video_detector("C:\\Users\\Gear\\Desktop\\new_cut_video\\แม่ไม้ 15 ท่า\\11_ นาคาบิดหาง (Subclip #3).mp4", 2)
detected_input = Video_detector("C:\\Users\\Gear\\Desktop\\new_cut_video\\แม่ไม้ 15 ท่า\\11_ นาคาบิดหาง (Subclip #3).mp4", 2)

for i,j in zip(detected_model, detected_input):

    result_model = landmark_formatter(i[0])[0]
    result_input = landmark_formatter(j[0])[0]

 