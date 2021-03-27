import cv2
from mtcnn import MTCNN
import math
import keras
from keras.models import load_model
import numpy as np



video = cv2.VideoCapture(0)

cv2.namedWindow("test")

#facedetector = cv2.CascadeClassifier("data\haarcascade_frontalface_default.xml")
#eyedetector = cv2.CascadeClassifier("data\haarcascade_eye_tree_eyeglasses.xml")

mtc = MTCNN()

model = load_model("model.h5", compile=False)

def check_eye(eye, threshold=0.5):
    eye_resized = cv2.resize(eye, (24, 24))
    X = np.reshape([eye_resized / 255], (-1, 24, 24, 1))
    if model.predict(X) > threshold:
        return "open", True
    else:
        return "close", False

try:
    is_open_right = False
    is_open_left = False


    while True:
        if not video.isOpened():
            continue

        ret, frame = video.read()

        if not ret:
            continue

        faces = mtc.detect_faces(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        for face in faces:
            (x, y, w, h) = face["box"]
            landmarks = face["keypoints"]
            (le_x, le_y) = (int(x) for x in landmarks["left_eye"])

            (re_x, re_y) = (int(x) for x in landmarks["right_eye"])

            distance = int(0.32*math.sqrt(math.pow(le_x - re_x, 2) + math.pow(re_y - le_y, 2)))
            print(distance)

            frame = cv2.rectangle(frame, (le_x - distance, le_y-distance), (le_x + distance, le_y + distance), (0, 255, 0),2)
            frame = cv2.rectangle(frame, (re_x - distance, re_y - distance), (re_x + distance, re_y + distance), (0, 255, 0), 2)

            left_eye = gray[le_y-distance: le_y+distance, le_x - distance : le_x + distance]
            right_eye = gray[re_y - distance: re_y + distance, re_x - distance: re_x + distance]

            if len(left_eye) != 0:
                result, bval = check_eye(left_eye)
                color = (0, 255, 0)
                if is_open_left and not bval:
                    color = (0, 0, 255)
                is_open_left = bval
                frame = cv2.putText(frame, result, (le_x - distance, le_y - distance), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    color, 1, 2)

            if len(right_eye) != 0:
                result, bval = check_eye(right_eye)
                color = (0, 255, 0)
                if is_open_right and not bval:
                    color = (0, 0, 255)

                is_open_right = bval

                frame = cv2.putText(frame, result, (re_x - distance, re_y - distance), cv2.FONT_HERSHEY_SIMPLEX, 1, color
                                    , 1, 2)

            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("test", frame)

        if cv2.waitKey(2) == ord("x"):
            break
finally:
    video.release()
    cv2.destroyAllWindows()