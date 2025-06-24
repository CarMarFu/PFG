import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import cv2
import os
from pathlib import Path


def image_recognition_get_csv(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo de video {file_path} no existe.")

    MODEL_PATH = "./hand_landmarker.task"
    FILE_NAME = Path(file_path).stem

    # initialize image recognition model
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options, running_mode=vision.RunningMode.VIDEO, num_hands=1
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    # import video and properties
    video_data = cv2.VideoCapture(file_path)
    fps = int(video_data.get(cv2.CAP_PROP_FPS))
    width = int(video_data.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_data.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = "./VIDEO_Export/" + FILE_NAME + "_processed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    coord_data = []
    frame_id = 0

    while video_data.isOpened():
        ret, frame = video_data.read()
        if not ret:
            break

        timestamp_ms = int((frame_id / fps) * 1000)
        frame_id += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
                for i, landmark in enumerate(hand_landmarks):
                    coord_data.append(
                        {
                            "frame": frame_id,
                            "hand_id": hand_idx,
                            "landmark_id": i,
                            "x": landmark.x,
                            "y": landmark.y,
                            "z": landmark.z,
                        }
                    )

                # draw a video of model seen
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    landmark_list_to_landmark_proto(hand_landmarks),
                    mp.solutions.hands.HAND_CONNECTIONS,
                )

        out.write(frame)

    video_data.release()
    out.release()
    cv2.destroyAllWindows()

    coord_dataframe = pd.DataFrame(coord_data)
    csv_path = "./VIDEO_Export/" + FILE_NAME + "_processed.csv"

    coord_dataframe.to_csv(csv_path, index=False)
    # returns the csv too
    return coord_dataframe


# needed for drawing in video sample
def landmark_list_to_landmark_proto(landmarks):
    landmark_proto = landmark_pb2.NormalizedLandmarkList()
    for lm in landmarks:
        landmark_proto.landmark.add(x=lm.x, y=lm.y, z=lm.z)
    return landmark_proto
