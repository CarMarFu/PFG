import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from pathlib import Path


def combine_video_emg_csv(emg_path, csv_path):
    # import and pivot video data
    video_data = pd.read_csv(csv_path)
    video_data = pivot_csv_table(video_data)
    video_data = smooth_video_coord(video_data)
    video_data = get_hand_angles(video_data)

    # import and process emg data
    emg_data = emg_import(emg_path)
    emg_data = trimm_emg(emg_data, video_data)
    emg_data = reshape_emg(emg_data, video_data)

    # create dataframe with all data
    X = pd.concat(
        [
            video_data.reset_index(drop=True),
            emg_data,
        ],
        axis=1,
    )

    FILE_NAME = Path(csv_path).stem
    csv_combined_path = "./MERGE_Export" + FILE_NAME + "_COMBINED.csv"
    X.to_csv(csv_combined_path, index=False)
    return


## VIDEO ###################################


def pivot_csv_table(video_data):
    video_data_wide = video_data.pivot_table(
        index="frame", columns="landmark_id", values=["x", "y", "z"]
    )

    # flatten multi-index columns
    video_data_wide.columns = [
        f"{coord}_lm{lm}" for coord, lm in video_data_wide.columns
    ]
    video_data_wide.reset_index(inplace=True)
    return video_data_wide


def smooth_video_coord(video_data, WINDOW=33, ORDER=2):
    # Make a copy of the original DataFrame
    video_data_filtered = video_data.copy()

    # Loop through all columns
    for col in video_data.columns:
        # Check if it's a coordinate column like x_lm#
        if col.startswith(("x_lm", "y_lm", "z_lm")) and video_data[col].dtype != "O":
            if len(video_data[col]) >= WINDOW:
                video_data_filtered[col] = savgol_filter(video_data[col], WINDOW, ORDER)
            else:
                video_data_filtered[col] = video_data[col]  # fallback if too short

    return video_data_filtered


def get_hand_angles(video_data):
    angle_columns = [
        "frame",
        "alfa_I",
        "alfa_II",
        "beta_I",
        "beta_II",
        "beta_III",
        "gamma_I",
        "gamma_II",
        "gamma_III",
        "lambda_I",
        "lambda_II",
        "lambda_III",
        "epsilon_I",
        "epsilon_II",
        "epsilon_III",
    ]

    list_points = [
        [1, 2, 3],  # alfa_I
        [2, 3, 4],  # alfa_II
        [5, 6, 7],  # beta_I
        [6, 7, 8],  # beta_II
        [0, 5, 6],
        [9, 10, 11],  # gamma_I
        [10, 11, 12],  # gamma_II
        [0, 9, 10],
        [13, 14, 15],  # lambda_I
        [14, 15, 16],  # lambda_II
        [0, 13, 14],
        [17, 18, 19],  # epsilon_I
        [18, 19, 20],  # epsilon_II
        [0, 17, 18],
    ]

    max_frames = video_data["frame"].max()
    angle_df = pd.DataFrame(columns=angle_columns)

    for frame in range(max_frames):
        frame_data = video_data[video_data["frame"] == frame + 1]
        angle_df.at[frame, "frame"] = frame + 1

        for i, point in enumerate(list_points):
            P1 = np.array(
                [
                    frame_data[f"x_lm{point[0]}"],
                    frame_data[f"y_lm{point[0]}"],
                    frame_data[f"z_lm{point[0]}"],
                ]
            )
            P2 = np.array(
                [
                    frame_data[f"x_lm{point[1]}"],
                    frame_data[f"y_lm{point[1]}"],
                    frame_data[f"z_lm{point[1]}"],
                ]
            )
            P3 = np.array(
                [
                    frame_data[f"x_lm{point[2]}"],
                    frame_data[f"y_lm{point[2]}"],
                    frame_data[f"z_lm{point[2]}"],
                ]
            )

            if all(
                arr.size != 0 for arr in [P1, P2, P3]
            ):  # evitar errores si faltan puntos
                angle_df.iat[frame, i + 1] = angle_between(P1, P2, P3)
            else:
                angle_df.iat[frame, i + 1] = np.nan  # por si faltan landmarks

    video_data_angles = pd.merge(video_data, angle_df, on="frame", how="left")

    return video_data_angles


def angle_between(P1, P2, P3):  # sacar ángulo con el arccoseno
    v1 = P2 - P1
    v2 = P3 - P2

    v1_u = unit_vector(v1).flatten()  # aplsatamos para que se pueda hacer dot product
    v2_u = unit_vector(v2).flatten()
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))  # output en rad


def unit_vector(vector):  # calcular vector unitario
    return vector / np.linalg.norm(vector)


## EMG ###################################


def emg_import(file_path, NUM_CHANNELS=5):
    data_emg = np.fromfile(file_path, dtype=np.float32)
    # data vectorized
    data_emg = data_emg.reshape((NUM_CHANNELS, data_emg.size // NUM_CHANNELS))

    return data_emg


def trimm_emg(emg_data, video_data, FPS=32, FS=1024):
    # check if trimming required and propose automatic trimming
    expected_ratio = FS / FPS
    actual_ratio = emg_data.shape[1] / video_data["frame"].nunique()
    if np.isclose(actual_ratio, expected_ratio) and actual_ratio.is_integer():
        print("INFO: The datasets are aligned with a natural number relation.")
        emg_data_trimmed = emg_data
    else:
        print(
            """WARNING: The datasets are misaligned or have fractional relation.
            Manual trimming is recommended, but some will be done
             by deleting the overflow at end of EMG data."""
        )
        duration = video_data["frame"].nunique() / FPS
        expected_emg_samples = int(duration * FS)
        emg_data_trimmed = emg_data[:, :expected_emg_samples]
    return emg_data_trimmed


def reshape_emg(emg_data, video_data, FPS=32, FS=1024, NUM_CHANNELS=5):
    num_frames = video_data["frame"].nunique()
    samples_per_frame = int(FS / FPS)

    columns = [f"emg_{i}_ch_{j}" for j in range(1, 6) for i in range(32)]
    data = np.zeros((num_frames, samples_per_frame * NUM_CHANNELS))
    for k in range(num_frames):
        for j in range(NUM_CHANNELS):
            start = k * samples_per_frame
            end = start + samples_per_frame
            data[k, j * samples_per_frame : (j + 1) * samples_per_frame] = emg_data[
                j, start:end
            ]

    emg_data_reshaped = pd.DataFrame(data, columns=columns)
    return emg_data_reshaped
