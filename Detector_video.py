import mediapipe as mp
import numpy as np
import cv2
from draw_landmarks import draw_landmarks_on_image

model_path = 'pose_landmarker_full.task'

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_poses= 2
    )

landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

def Video_detector(video_path):
# Use OpenCV’s VideoCapture to start capturing from the webcam.
    cap = cv2.VideoCapture(video_path)
    frame_timestamp_ms = 0

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
        pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        annotated_image = draw_landmarks_on_image(mp_image, pose_landmarker_result)
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("OUTPUT", annotated_image)
        frame_timestamp_ms += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

Video_detector(0)