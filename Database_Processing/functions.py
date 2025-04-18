import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from scipy.signal import iirnotch, filtfilt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import cv2
import os


## FUNCTIONS RELATED TO EMG
def emg_import(rec_id):
    file_path = "./EMG_data/" + rec_id + "_Raw Data"
    data_emg = np.fromfile(file_path + ".dat", dtype=np.float32)
    # data vectorized
    data_emg = data_emg.reshape((10, data_emg.size // 10))
    # matrix 10xn holds 10 channels of emg
    data_emg = emg_notch_filter(data_emg)

    markers_tree = ET.parse(file_path + ".xmrk")
    markers_root = markers_tree.getroot()
    NAMESPACE = {"ns": "http://www.brainproducts.com/MarkerSet"}
    markers_emg_all = [
        int(marker.find("ns:Position", NAMESPACE).text)
        for marker in markers_root.findall("ns:Markers/ns:Marker", NAMESPACE)
    ]

    markers_emg = [markers_emg_all[1], markers_emg_all[-1]]

    data_emg_trimmed = data_emg[markers_emg[0] : markers_emg[1], :]
    return data_emg_trimmed


def emg_notch_filter(emg_data):
    FS = 500
    F0 = 50
    Q = 30
    b, a = iirnotch(F0, Q, FS)

    emg_data_filtered = np.zeros_like(emg_data.size)
    for i in range(emg_data_filtered.size[0]):
        emg_data_filtered[i, :] = np.array(filtfilt(b, a, emg_data[i, :]))
    return emg_data_filtered


def emg_compute_bipolar(emg_data):
    emg_data_bipolar = np.zeros((5, emg_data.size[1]))
    for i in range(5):
        emg_data_bipolar[i, :] = emg_data[2 * i, :] - emg_data[2 * i + 1, :]
    return emg_data_bipolar


## FUNCTIONS RELATED TO IMAGE RECOGNITION
def image_recognition_get_csv(video_id):
    file_path = "./video_data/" + video_id + ".mp4"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo de video {file_path} no existe.")

    MODEL_PATH = "./hand_landmarker.task"

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
    output_path = "./video_data/" + video_id + "_processed.mp4"
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

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
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
    csv_path = "./video_data/" + video_id + ".csv"
    coord_dataframe.to_csv(csv_path, index=False)
    # returns the csv too
    return coord_dataframe


# needed for drawing in video sample
def landmark_list_to_landmark_proto(landmarks):
    landmark_proto = landmark_pb2.NormalizedLandmarkList()
    for lm in landmarks:
        landmark_proto.landmark.add(x=lm.x, y=lm.y, z=lm.z)
    return landmark_proto


def combine_video_emg_csv(id):
    # check for csv and import
    csv_path = "./video_data/" + id + ".csv"
    if not os.path.exists(csv_path):
        print(
            f"Warning: El archivo de csv {csv_path} no existe. Generando uno nuevo. Esto puede tardar unos minutos."
        )
        video_csv_data = image_recognition_get_csv(id)
    else:
        video_csv_data = pd.read_csv(csv_path)

    # process emg data as it is imported
    emg_data = emg_compute_bipolar(emg_notch_filter(emg_import(id)))

    # add columns with pulses at end of video data
    num_video_frames = len(video_csv_data)
    num_emg_pulses = emg_data.shape[1]
    num_emg_channels = emg_data.shape[0]
    if num_emg_pulses % num_video_frames != 0:
        print(
            f"Warning: El número de pulsos de EMG ({num_emg_pulses}) no es múltiplo del número de frames del vídeo ({num_video_frames})"
        )

    num_pulses_per_frame = num_emg_pulses // num_video_frames
    for channel in range(num_emg_channels):
        for pulse in range(num_pulses_per_frame):
            col_name = f"ch{channel + 1}_pulse{pulse + 1}"
            video_csv_data[col_name] = pd.NA

    for frame in range(num_video_frames):
        for channel in range(num_emg_channels):
            for pulse in range(num_pulses_per_frame):
                value = emg_data[channel, frame * num_pulses_per_frame + pulse]
                col_name = f"ch{channel + 1}_pulse{pulse + 1}"
                video_csv_data.at[frame, col_name] = value

    csv_path = "./video_data/" + id + "_combined_EMG.csv"
    video_csv_data.to_csv(csv_path, index=False)
    return
