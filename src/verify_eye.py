from typing import Tuple
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import time
import numpy as np
from test import alert
import os
import json
from test import alert

dataPath = "../../../data/10.114.77.242/Training_Evaluation_Dataset/Training Dataset"

# dataSetYawn = {
#     "path": dataPath,
#     "data": {
#         "glasses": [],
#         "night_noglasses": [],
#         "nightglasses": [],
#         "noglasses": [],
#         "sunglasses": []
#     }
# }

# number = os.listdir(dataPath)

# for id in number:
#     scenarios = os.listdir(os.path.join(dataPath, id))
#     for scenario in scenarios:
#         yawningVideo = os.path.join(dataPath, id, scenario, 'sleepyCombination.avi')
#         if os.path.exists(yawningVideo):
#             dataSetYawn["data"][scenario].append(id)

# print(json.dumps(dataSetYawn, indent=4))
# with open('eye_data_manager.json', "w") as f:
#     json.dump(dataSetYawn, f, indent=4)


# video = os.path.join(dataPath, "001", "noglasses", "sleepyCombination.avi")
# label = os.path.join(dataPath, "001", "noglasses", "001_sleepyCombination_eye.txt")

# print(video)
# print(os.path.exists(video))

# cap = cv2.VideoCapture(video)
# f = open(label, "r")
# label_str = f.read()

# numFrame = 0
# while True:
#     ret, frame = cap.read()
#     frame = imutils.resize(frame, width=450)

#     cv2.putText(frame, f'label: {label_str[numFrame]}',        \
#                 (frame.shape[1] - 120, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3,  \
#                 (0, 0, 255), 1)
#     cv2.imshow("Frame", frame)
#     numFrame+=1
#     if numFrame == len(label_str):
#         break
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord("q"):
#         break
# cv2.destroyAllWindows()
# cap.release()


detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("../model/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]


with open('eye_data_manager.json', "r") as f:
    dataSetEye = json.load(f)

for scenario in dataSetEye["data"]:
    print(f"# ---------------------------------------Report for {scenario}---------------------------------------- #")
    path_videos = []
    path_labels = []

    [path_videos.append(os.path.join(dataSetEye["path"], number, scenario, 'slowBlinkWithNodding.avi')) for number in dataSetEye["data"][scenario]]
    [path_labels.append(os.path.join(dataSetEye["path"], number, scenario, f'{number}_slowBlinkWithNodding_eye.txt')) for number in dataSetEye["data"][scenario]]

    if (scenario == "night_noglasses" or scenario == "night_glasses"):
        video_fps = 15
    else:
        video_fps = 30

    for index_path, path_video in enumerate(path_videos):
        cap = cv2.VideoCapture(path_video)
        f = open(path_labels[index_path], "r")
        label_str = f.read()
        totalFrame = len(label_str)

        closeEyeArray = []
        closedEyeTime = 0
        lastCloseEye = 0.5
        close_eye_time_sleepy = []
        current_frame = 0
        while current_frame < totalFrame:
            ret, frame = cap.read()
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            smooth = cv2.GaussianBlur(gray, (125, 125), 0)
            division = cv2.divide(gray, smooth, scale=255)
            subjects = detect(division, 0)

            print(f"\rCurrent: {path_video.split('Dataset')[2]} {current_frame+1}/{totalFrame}", end="")

            for subject in subjects:
                (x, y, w, h) = face_utils.rect_to_bb(subject)
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 1)
                shape = predict(gray, subject)
                shape = face_utils.shape_to_np(shape)
                predictValue = alert(shape, eyeOnly=True, yawnOnly=False)

                for index, closeEye in enumerate(closeEyeArray):
                    closeEyeArray[index] = closeEyeArray[index] + float(float(1)/(video_fps))

                for index, closeEyeCal in enumerate(closeEyeArray):
                    if closeEyeCal >= 30:
                        closeEyeArray.remove(closeEyeCal)

                lastCloseEye = lastCloseEye + float(float(1)/(video_fps))
                if predictValue % 2:
                    # 1 or 3: mean closing eye
                    if lastCloseEye >= 0.5:
                        lastCloseEye = 0
                        closeEyeArray.append(0)
                if label_str[current_frame] == '1':
                    close_eye_time_sleepy.append(len(closeEyeArray))
            current_frame += 1
            cv2.putText(frame, f'label: {label_str[current_frame-1]}',                              \
                    (frame.shape[1] - 120, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, \
                    (0, 0, 255), 2)
            cv2.putText(frame, f'pred: {predictValue % 2}',                              \
                    (frame.shape[1] - 120, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, \
                    (0, 0, 255), 2)
            cv2.putText(frame, f'closed: {len(closeEyeArray)} times',                              \
                    (frame.shape[1] - 120, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, \
                    (0, 0, 255), 2)

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        aver = round(float(sum(close_eye_time_sleepy))/len(close_eye_time_sleepy),2)
        print(f" - Report: {aver} {close_eye_time_sleepy}")