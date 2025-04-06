import freenect
import cv2 as cv
 
# Funcion para capturar un fotograma del video del Kinect
def get_video():
    frame, _ = freenect.sync_get_video()
    return cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    
# Bucle principal para mostrar el video en tiempo real
while True:
    frame = get_video()
    cv.imshow('Kinect Video', frame)
    
    key = cv.waitKey(10)
    if key == ord('q'): # Cerrarel frame si se pulsa el boton 'q'
        break
    
cv.destroyAllWindows()
