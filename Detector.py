import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2 
import datetime

model_path = 'pose_landmarker_lite.task'

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

# Create a pose landmarker instance with the live stream mode:
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
  print(len(result.pose_landmarks))
  annotated_image = draw_landmarks_on_image(output_image.numpy_view(), result)
  annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
  cv2.imshow("OUTPUT", annotated_image)
  cv2.waitKey(1)

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_poses= 2,
    result_callback=print_result)

detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)


with PoseLandmarker.create_from_options(options) as landmarker:
  # Use OpenCV’s VideoCapture to start capturing from the webcam.
  cap = cv2.VideoCapture("C:\\Users\\Gear\\Downloads\\5_ ยอเขาพระสุเมรุ (Subclip).mp4")
  frame_time_stamp = 1

  # Create a loop to read the latest frame from the camera using VideoCapture#read()
  if not cap.isOpened():
    print("Cannot open camera")
    exit()
  while True:

    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  

    # Convert the frame received from OpenCV to a MediaPipe’s Image object.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    # The landmarker is initialized. Use it here.
    annotated_image = landmarker.detect_async(mp_image, frame_time_stamp)
    frame_time_stamp += 1
    print(frame_time_stamp)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()