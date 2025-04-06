import freenect
import cv2
import serial
import time

arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
time.sleep(2)

arduino.write(b'H')  # Enviar comando al Arduino
time.sleep(1)  # Espera 1 segundo
arduino.write(b'L')
time.sleep(1)

def get_video():
    frame, _ = freenect.sync_get_video()
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# Hace falta ajustar FPS a sampling_frequency
output_video = cv2.VideoWriter('kinect_video.mp4', fourcc, 30.0, (640, 480))

while True:
    frame = get_video()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    output_video.write(frame)
    cv2.imshow('Kinect Video', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

output_video.release()
cv2.destroyAllWindows()
arduino.close()

