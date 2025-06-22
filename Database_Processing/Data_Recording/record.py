import freenect
import cv2
import serial
import time
import os


def get_next_filename(folder="./VIDEO_Raw/", prefix="", extension=".mp4", digits=6):
    os.makedirs(folder, exist_ok=True)
    i = 1
    while True:
        filename = f"{prefix}{i:0{digits}d}{extension}"
        full_path = os.path.join(folder, filename)
        if not os.path.exists(full_path):
            return full_path
        i += 1


def get_video():
    frame, _ = freenect.sync_get_video()
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


arduino = serial.Serial("/dev/ttyACM0", 9600, timeout=1)
time.sleep(2)

output_filename = get_next_filename()
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_video = cv2.VideoWriter(output_filename, fourcc, 50.0, (640, 480))

arduino.write(b"H")
time.sleep(1)
arduino.write(b"L")

while True:
    frame = get_video()
    output_video.write(frame)
    cv2.imshow("Kinect Video", frame)

    # exit with q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

output_video.release()
arduino.write(b"H")
time.sleep(1)
arduino.write(b"L")
cv2.destroyAllWindows()
arduino.close()
