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

# dataPath = "../../../data/10.114.77.242/Training_Evaluation_Dataset/Training Dataset"

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
#         yawningVideo = os.path.join(dataPath, id, scenario, 'yawning.avi')
#         if os.path.exists(yawningVideo):
#             dataSetYawn["data"][scenario].append(id)
#             # print(yawningVideo)

# print(json.dumps(dataSetYawn, indent=4))
# with open('yawning_data_manager.json', "w") as f:
#     json.dump(dataSetYawn, f, indent=4)


with open('yawning_data_manager.json', "r") as f:
    dataSetYawn = json.load(f)

for scenario in dataSetYawn["data"]:
    print(f"# ---------------------------------------Report for {scenario}---------------------------------------- #")
    path_videos = []
    path_labels = []
    [path_videos.append(os.path.join(dataSetYawn["path"], number, scenario, 'yawning.avi')) for number in dataSetYawn["data"][scenario]]
    [path_labels.append(os.path.join(dataSetYawn["path"], number, scenario, f'{number}_yawning_mouth.txt')) for number in dataSetYawn["data"][scenario]]

    hog_face_detector = dlib.get_frontal_face_detector()
    dlib_facelandmark = dlib.shape_predictor("../model/shape_predictor_68_face_landmarks.dat")

    mouthThreshes = []
    for i in range(0, 30):
        mouthThreshes.append(round(0.1 + i*0.02, 2))
    totalFrame = 0
    predictTruesTotal = []
    for i in range(0, 30):
        predictTruesTotal.append(0)

    for index_path, path_video in enumerate(path_videos):
        cap = cv2.VideoCapture(path_video)
        f = open(path_labels[index_path], "r")
        label_str = f.read()
        totalFrame += len(label_str)

        numberOfFrame = 0
        predictTrues = []
        for i in range(0, 30):
            predictTrues.append(0)

        print(f"Report {index_path+1}/{len(path_videos)} {path_video.split('Dataset')[2]}:")
        while True:
            ret, frame = cap.read()
            numberOfFrame += 1
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            smooth = cv2.GaussianBlur(gray, (125, 125), 0)
            division = cv2.divide(gray, smooth, scale=255)

            faces = hog_face_detector(division, 0)
            for face in faces:
                shape = dlib_facelandmark(gray, face)
                shape = face_utils.shape_to_np(shape)
                predictValues = [alert(shape, eyeOnly=False, yawnOnly=True, mouthThresh=mouthThresh) for mouthThresh in mouthThreshes]
                for index_predict, predictValue in enumerate(predictValues):
                    # predictValue = 0 or 2
                    # label_str[numberOfFrame-1] = 0 or 1 or 2
                    if (predictValue==0) == ((int(label_str[numberOfFrame-1])) == 0):
                        predictTrues[index_predict] += 1

            print(f'\r\tExecuting: {numberOfFrame}/{len(label_str)}',end='')

            if len(label_str) == numberOfFrame:
                break
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        for index, predictTrue in enumerate(predictTrues):
            predictTruesTotal[index] += predictTrues[index]
            predictTrues[index] = round(100 * float(predictTrues[index]) / len(label_str), 2)

        print(f'\n\tPredict true: {predictTrues}')

    print(f"\n\n Total: {totalFrame} frames")
    for index, predictTrue in enumerate(predictTruesTotal):
        print(f'Mouth thresh: {mouthThreshes[index]}: {round(100 * float(predictTrue) / totalFrame, 2)}%')
