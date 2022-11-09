from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import time
import numpy as np

# Thresh variables
eyeThresh = 0.20
mouthThresh = 0.28
closeEyesCheck = 2   # Seconds

# Time variables
currentTime = time.time()
previousTime = time.time()
runningTime = 0
closedEyeTime = 0
yawnCalArray = []
closeEyeArray = []
lastYawn = 5
lastCloseEye = 3

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)

    return round(ear, 4)


def mouth_aspect_ratio(mouth):
    topLip = mouth[2:5]
    topLip = np.concatenate((topLip, mouth[13:16]))

    lowLip = mouth[8:11]
    lowLip = np.concatenate((lowLip, mouth[17:20]))

    topMean = np.mean(topLip, axis=0)
    lowMean = np.mean(lowLip, axis=0)

    dist = distance.euclidean(topMean, lowMean)
    openRate = float(dist) / distance.euclidean(mouth[0], mouth[6])

    return round(openRate, 4)


def show_all_point(thing, printNumber=False, color=(255, 255, 0)):
    for i, point in enumerate(thing):
        if not printNumber:
            frame[point[1], point[0]] = list(color)
        else:
            cv2.putText(frame, str(i), (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.2, color, 1)


def alert(shape=[], eyeOnly=False, yawnOnly=False, eyeThresh=eyeThresh, mouthThresh=mouthThresh):
    """
    Description:
        - Detect closed eye and yawn alert.

    Param:
        - shape: shape of face detected
        - eyeOnly: Only detect closed eye alert
        - yawnOnly: Only detect yawn alert

    Return: int
        - 0: No alert
        - 1: closing eye
        - 2: yawning
        - 3: both yawning and closing eye
    """

    retValue = 0
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    mouth = shape[mStart:mEnd]

    # show_all_point(leftEye, printNumber=False)
    # show_all_point(rightEye, printNumber=False)
    # show_all_point(mouth, printNumber=False)

    if not yawnOnly:
        # Detect closed eye
        leftEyeEar = eye_aspect_ratio(leftEye)
        rightEyeEar = eye_aspect_ratio(rightEye)
        ear = (leftEyeEar + rightEyeEar) / 2
        if ear < eyeThresh:
            retValue += 1

    if not eyeOnly:
        # Detect yawn
        openMouthRate = mouth_aspect_ratio(mouth)
        if openMouthRate > mouthThresh:
            retValue += 2

    # cv2.putText(frame, f'EAR Left eye: {eye_aspect_ratio(leftEye)}',        \
    #             (frame.shape[1] - 120, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3,  \
    #             (0, 0, 255), 1)
    # cv2.putText(frame, f'EAR Right eye: {eye_aspect_ratio(rightEye)}',      \
    #             (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3,  \
    #             (0, 0, 255), 1)
    # cv2.putText(frame, f'open Mouth: {openMouthRate}',                      \
    #             (frame.shape[1] - 120, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.3,  \
    #             (0, 0, 255), 1)

    return retValue

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("../model/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

# Take video input
cap = cv2.VideoCapture(0)

if __name__ == '__main__':
    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        smooth = cv2.GaussianBlur(gray, (125, 125), 0)
        division = cv2.divide(gray, smooth, scale=255)
        subjects = detect(division, 0)

        currentTime = time.time()
        runningTime = round(runningTime + currentTime - previousTime, 2)
        fps = int(1 / (currentTime - previousTime))

        for subject in subjects:
            (x, y, w, h) = face_utils.rect_to_bb(subject)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 1)
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            predictValue = alert(shape, eyeOnly=False, yawnOnly=False)

            ################################################################################
            # Handle Blink eye to much
            for index, closeEye in enumerate(closeEyeArray):
                closeEyeArray[index] = closeEyeArray[index] + currentTime - previousTime

            for index, closeEyeCal in enumerate(closeEyeArray):
                if closeEyeCal >= 60:
                    closeEyeArray.remove(closeEyeCal)

            lastCloseEye = lastCloseEye + currentTime - previousTime

            ################################################################################
            # Handle Close eye alert
            if predictValue % 2:
                # 1 or 3: mean closing eye
                if lastCloseEye >= 3:
                    lastCloseEye = 0
                    closeEyeArray.append(0)

                closedEyeTime = closedEyeTime + currentTime - previousTime
                if closedEyeTime >= closeEyesCheck:
                    cv2.putText(frame, "CLOSED EYES ALERT!",            \
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,\
                                (0, 0, 255), 2)
            else:
                closedEyeTime = 0

            if len(closeEyeArray) >= 10 and closedEyeTime == 0:
                cv2.putText(frame, "BLINK EYE TOO MUCH ALERT!",         \
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7,    \
                            (0, 0, 255), 2)

            ################################################################################
            # Handle yawn too much
            for index, yawnCal in enumerate(yawnCalArray):
                yawnCalArray[index] = yawnCalArray[index] + currentTime - previousTime

            for index, yawnCal in enumerate(yawnCalArray):
                if yawnCal >= 60:
                    yawnCalArray.remove(yawnCal)

            lastYawn = lastYawn + currentTime - previousTime

            ################################################################################
            # Handle yawn alert
            if predictValue >= 2 and lastYawn >= 5:
                # 2 or 3: mean yawning
                lastYawn = 0
                yawnCalArray.append(0)

            if len(yawnCalArray) >=3:
                cv2.putText(frame, "YAWN ALERT!",                       \
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7,    \
                            (0, 0, 255), 2)


        cv2.putText(frame, f'closed Eye Time: {closedEyeTime}',                 \
                    (frame.shape[1] - 120, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.3,  \
                    (0, 0, 255), 1)
        cv2.putText(frame, f'blink eye in last 60s: {len(closeEyeArray)}',      \
                    (frame.shape[1] - 120, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.3,  \
                    (0, 0, 255), 1)
        cv2.putText(frame, f'yawn in last 60s: {len(yawnCalArray)}',            \
                    (frame.shape[1] - 120, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.3,  \
                    (0, 0, 255), 1)
        cv2.putText(frame, f'FPS: {fps}',                                                       \
                    (frame.shape[1] - 120, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, \
                    (0, 0, 255), 2)
        cv2.putText(frame, f'Time: {runningTime}',                                              \
                    (frame.shape[1] - 120, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, \
                    (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        previousTime = currentTime
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()
