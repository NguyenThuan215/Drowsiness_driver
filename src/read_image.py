import os
from imutils import face_utils
import dlib
import cv2
from test import alert
from test import mouth_aspect_ratio

listImage = []
pathYawn = ["../../../data/yawn_eye_dataset/dataset_new/train/yawn",
            "../../../data/yawn_eye_dataset/dataset_new/test/yawn"]

pathNoYawn = ["../../../data/yawn_eye_dataset/dataset_new/train/no_yawn",
              "../../../data/yawn_eye_dataset/dataset_new/test/no_yawn"]

listImageYawn = []
listImageNoYawn = []

for path in pathYawn:
    [listImageYawn.append(path + '/' + image) for image in os.listdir(path)]

for path in pathNoYawn:
    [listImageNoYawn.append(path + '/' + image) for image in os.listdir(path)]

print(f"Number of image in yawn dataset: {len(listImageYawn)}")
print(f"Number of image in no-yawn dataset: {len(listImageNoYawn)}")
print(f"Total: {len(listImageYawn) + len(listImageNoYawn)}")

# ------------------------------------Calculate accuracy of face detection-------------------------------- #
print("\nReport face detection: ")
numberOfFace = 0
listImage.extend(listImageYawn)
listImage.extend(listImageNoYawn)
hog_face_detector = dlib.get_frontal_face_detector()

for index, imagePath in enumerate(listImage):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    smooth = cv2.GaussianBlur(gray, (125, 125), 0)
    division = cv2.divide(gray, smooth, scale=255)

    faces = hog_face_detector(division, 0)
    print(f'\rExecuting: {index+1}/{len(listImage)}', end='')
    for face in faces:
        numberOfFace += 1

print(f'\nNumber of face detected: {numberOfFace}')
print(f"Accuracy face detector: {round(100 * float(numberOfFace) / len(listImage), 2)}")

# ------------------------------------Calculate accuracy of yawn detection-------------------------------- #
print("\nReport yawn detection: ")
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("../model/shape_predictor_68_face_landmarks.dat")

(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

mouthThreshes = [0.28]
for i in range(0, 30):
    mouthThreshes.append(round(0.1 + i*0.02, 2))


for mouthThresh in mouthThreshes:
    numberOfFaceYawn = 0
    predictTrueYawn = 0
    for index, imagePath in enumerate(listImageYawn):
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        smooth = cv2.GaussianBlur(gray, (125, 125), 0)
        division = cv2.divide(gray, smooth, scale=255)

        faces = hog_face_detector(gray, 0)
        for face in faces:
            numberOfFaceYawn += 1
            shape = dlib_facelandmark(gray, face)
            shape = face_utils.shape_to_np(shape)
            predictValue, openMouthRate = alert(shape, eyeOnly=False, yawnOnly=True, mouthThresh=mouthThresh)
            if predictValue == 2:
                predictTrueYawn += 1
                for point in shape:
                    image[point[1], point[0]] = [255, 255, 0]
                cv2.putText(image, str(openMouthRate), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [255, 255, 0], 1)
                cv2.imwrite("../../../data/yawn_eye_dataset/dataset_new/result/" + "yawn_true" + imagePath.split("/")[-1], image)
            else:
                for point in shape:
                    image[point[1], point[0]] = [255, 255, 0]
                cv2.imwrite("../../../data/yawn_eye_dataset/dataset_new/result/" + "yawn_false" + imagePath.split("/")[-1], image)

    print(f'Report with mouthThresh = {mouthThresh}')
    print(" Yawn:")
    print(f'    Number of face in yawn-dataset detected: {numberOfFaceYawn}')
    print(f'    Predict yawn true: {predictTrueYawn}')
    print(f'    Accuracy yawn detector: {round(100 * float(predictTrueYawn) / numberOfFaceYawn, 2)}%')

    numberOfFaceNoYawn = 0
    predictTrueNoYawn = 0
    for index, imagePath in enumerate(listImageNoYawn):
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        smooth = cv2.GaussianBlur(gray, (125, 125), 0)
        division = cv2.divide(gray, smooth, scale=255)

        faces = hog_face_detector(division, 0)
        for face in faces:
            numberOfFaceNoYawn += 1
            shape = dlib_facelandmark(gray, face)
            shape = face_utils.shape_to_np(shape)
            predictValue = alert(shape, eyeOnly=False, yawnOnly=True, mouthThresh=mouthThresh)
            if predictValue == 0:
                predictTrueNoYawn += 1

    print(" No-Yawn:")
    print(f'    Number of face in no-yawn-dataset detected: {numberOfFaceNoYawn}')
    print(f'    Predict no-yawn true: {predictTrueNoYawn}')
    print(f'    Accuracy no yawn detector: {round(100 * float(predictTrueNoYawn) / numberOfFaceNoYawn, 2)}%')
    print()

