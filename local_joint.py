import numpy as np

def extract_local_angle(pose_landmarker_result):
  r_wrist = cal_local_angle(pose_landmarker_result[20], pose_landmarker_result[16], pose_landmarker_result[14])
  r_elbow = cal_local_angle(pose_landmarker_result[16], pose_landmarker_result[14], pose_landmarker_result[12])
  r_shoulder = cal_local_angle(pose_landmarker_result[14], pose_landmarker_result[12], pose_landmarker_result[24])
  r_hip = cal_local_angle(pose_landmarker_result[12], pose_landmarker_result[24], pose_landmarker_result[26])
  r_knee = cal_local_angle(pose_landmarker_result[24], pose_landmarker_result[26], pose_landmarker_result[28])
  r_ankle = cal_local_angle(pose_landmarker_result[26], pose_landmarker_result[28], pose_landmarker_result[32])

  l_wrist = cal_local_angle(pose_landmarker_result[19], pose_landmarker_result[15], pose_landmarker_result[13])
  l_elbow = cal_local_angle(pose_landmarker_result[15], pose_landmarker_result[13], pose_landmarker_result[11])
  l_shoulder = cal_local_angle(pose_landmarker_result[13], pose_landmarker_result[11], pose_landmarker_result[23])
  l_hip = cal_local_angle(pose_landmarker_result[11], pose_landmarker_result[23], pose_landmarker_result[25])
  l_knee = cal_local_angle(pose_landmarker_result[23], pose_landmarker_result[25], pose_landmarker_result[27])
  l_ankle = cal_local_angle(pose_landmarker_result[25], pose_landmarker_result[27], pose_landmarker_result[31])

  return np.array([r_wrist, r_elbow, r_shoulder, r_hip, r_knee, r_ankle,
                   l_wrist, l_elbow, l_shoulder, l_hip, l_knee, l_ankle])


def cal_local_angle(point1, point2, point3):
  angle = np.rad2deg(np.arccos(np.dot(cal_norm_vec(point1, point2), cal_norm_vec(point3,point2))))

  return angle

def cal_norm_vec(point1,point2):
  vec = point2-point1
  vec_mag = np.linalg.norm(vec)
  vec_norm = vec/vec_mag

  return vec_norm


def distance_local_angle(ang1, ang2):

  return np.sum(np.where(np.abs(ang1-ang2) <= 180 , np.abs(ang1-ang2), 360-np.abs(ang1-ang2))) #0-(12*180=2160)