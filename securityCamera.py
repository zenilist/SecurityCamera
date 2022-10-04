'''Uses the opencv library to open camera and start recording security footage when a face is detected. The recording
stops once no face is detected.
 The program saves video files with current time of the system.'''

import cv2
import time
import datetime

capture = cv2.VideoCapture(0)

# trained algorithm for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")


recording = False
detection_stopped_time = None
timer_started = False
STOP_RECORDING = 5

frame_size = (int(capture.get(3)), int(capture.get(4)))
fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")

while True:
    p, frame = capture.read()

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayscale, 1.3, 5)
    bodies = face_cascade.detectMultiScale(grayscale, 1.3, 5)

    if len(faces) + len(bodies) > 0:
        if recording:
            timer_started = False
        else:
            recording = True
            timeNow = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv2.VideoWriter(f"securityFootage_{timeNow}.mp4", fourcc, 17, frame_size)
            print("Started recording")
    elif recording:
        if timer_started:
            if time.time() - detection_stopped_time >= STOP_RECORDING:
                recording = False
                timer_started = False
                out.release()
                print("Stopped Recording")
        else:
            timer_started = True
            detection_stopped_time = time.time()

    if recording:
        out.write(frame)

    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 2)

    cv2.imshow("Camera", frame)

    # when e is pressed the recording stops
    if cv2.waitKey(1) == ord('e'):
        break

out.release()
capture.release()
cv2.destroyAllWindows()
