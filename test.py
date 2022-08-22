from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import time
import numpy as np


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return round(ear, 4)


def cal_yawn(mouthShape):
    topLip = mouthShape[2:5]
    topLip = np.concatenate((topLip, mouthShape[13:16]))

    lowLip = mouthShape[8:11]
    lowLip = np.concatenate((lowLip, mouthShape[17:20]))

    topMean = np.mean(topLip, axis=0)
    lowMean = np.mean(lowLip, axis=0)

    dist = distance.euclidean(topMean, lowMean)
    openRate = float(dist) / distance.euclidean(mouthShape[0], mouthShape[6])

    return round(openRate, 4)


def show_all_point(thing, printNumber=False, color=(255, 255, 0)):
    for i, point in enumerate(thing):
        if not printNumber:
            frame[point[1], point[0]] = list(color)
        else:
            cv2.putText(frame, str(i), (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.2, color, 1)


eyeThresh = 0.20
yawnThresh = 0.75
flag = 0
frame_check = 20

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]
cap = cv2.VideoCapture(0)

print(face_utils.FACIAL_LANDMARKS_68_IDXS.keys())
ptime = 0

while True:
    cv2.waitKey(100)
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    ctime = time.time()
    fps = int(1 / (ctime - ptime))
    ptime = ctime
    cv2.putText(frame, f'FPS:{fps}', (frame.shape[1] - 120, frame.shape[0] - 20), cv2.FONT_HERSHEY_PLAIN, 2,
                (0, 200, 0), 3)

    for subject in subjects:
        (x, y, w, h) = face_utils.rect_to_bb(subject)
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 1)
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        leftEyeEar = eye_aspect_ratio(leftEye)
        rightEyeEar = eye_aspect_ratio(rightEye)
        ear = (leftEyeEar + rightEyeEar) / 2

        if ear < eyeThresh:
            flag += 1
            cv2.putText(frame, f'{flag}', (340, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            if flag >= frame_check:
                cv2.putText(frame, "CLOSE EYES ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            flag = 0

        openMouthRate = cal_yawn(mouth)
        cv2.putText(frame, f'open Mouth: {openMouthRate}', (340, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        if openMouthRate > yawnThresh:
            cv2.putText(frame, "YAWN ALERT!", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        show_all_point(leftEye)
        show_all_point(rightEye)
        show_all_point(mouth, printNumber=False)
        cv2.putText(frame, f'EAR Left eye: {eye_aspect_ratio(leftEye)}', (340, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (0, 0, 255), 1)
        cv2.putText(frame, f'EAR Right eye: {eye_aspect_ratio(rightEye)}', (340, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (0, 0, 255), 1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
cap.release()
