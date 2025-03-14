import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import cv2

model_path = './hand_landmarker.task'

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Cargar el video
video_path = './hand_rec_demo_video.mp4'
cap = cv2.VideoCapture(video_path)

# Lista para almacenar las coordenadas
data = []

# Obtener propiedades del video original
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Definir el codec y crear un VideoWriter para guardar el video procesado
output_path = "./video_procesado_demo.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Configurar MediaPipe Hands
with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
    frame_id = 0  # Contador de frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1  # Aumentar contador de frames
        print(f"Procesando frame {frame_id}")

        # Convertir BGR a RGB para MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar el frame con MediaPipe Hands
        results = hands.process(frame_rgb)

        # Extraer coordenadas si se detectan manos
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                for i, landmark in enumerate(hand_landmarks.landmark):
                    data.append({
                        "frame": frame_id,
                        "hand_id": hand_idx,
                        "landmark_id": i,
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z
                    })

        # Dibujar landmarks si se detectan manos
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Guardar el frame en el nuevo video
        out.write(frame)

# Liberar recursos
cap.release()
out.release()

# Guardar en CSV las coordenadas
df = pd.DataFrame(data)
csv_path = "./landmarks_demo.csv"
df.to_csv(csv_path, index=False)

print(f"Datos guardados en: {csv_path}")

print(f"Video procesado guardado en: {output_path}")