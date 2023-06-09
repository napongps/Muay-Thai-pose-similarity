import numpy as np

def find_limb(pose_landmarker_result):
  R_findex_ankle = pose_landmarker_result[32] - pose_landmarker_result[28]
  R_ankle_knee = pose_landmarker_result[28] - pose_landmarker_result[26]
  R_knee_hip = pose_landmarker_result[26] - pose_landmarker_result[24]
  R_hip_shoulder = pose_landmarker_result[24] - pose_landmarker_result[12]
  R_index_wrist = pose_landmarker_result[20] - pose_landmarker_result[16]
  R_wrist_elbow = pose_landmarker_result[16] - pose_landmarker_result[14]
  R_elbow_shoulder = pose_landmarker_result[14] - pose_landmarker_result[12]
  
  L_findex_ankle = pose_landmarker_result[31] - pose_landmarker_result[27]
  L_ankle_knee = pose_landmarker_result[27] - pose_landmarker_result[25]
  L_knee_hip = pose_landmarker_result[25] - pose_landmarker_result[23]
  L_hip_shoulder = pose_landmarker_result[23] - pose_landmarker_result[11]
  L_index_wrist = pose_landmarker_result[19] - pose_landmarker_result[15]
  L_wrist_elbow = pose_landmarker_result[15] - pose_landmarker_result[13]
  L_elbow_shoulder = pose_landmarker_result[13] - pose_landmarker_result[11]
  
  lshoulder_rshoulder = pose_landmarker_result[11] - pose_landmarker_result[12]
  lhip_rhip = pose_landmarker_result[23] - pose_landmarker_result[24]

  return np.array([R_findex_ankle, R_ankle_knee, R_knee_hip, R_hip_shoulder, R_index_wrist, R_wrist_elbow, R_elbow_shoulder,
          L_findex_ankle, L_ankle_knee, L_knee_hip, L_hip_shoulder, L_index_wrist, L_wrist_elbow, L_elbow_shoulder,
          lshoulder_rshoulder, lhip_rhip])

def cosine_distance(vec1,vec2):
  
  return np.sum(1 - np.sum(vec1*vec2, axis=1)/(np.linalg.norm(vec1, axis=1)*np.linalg.norm(vec2, axis=1))) # 0-24