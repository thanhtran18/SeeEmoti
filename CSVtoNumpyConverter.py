import cv2
import pandas as pd
import numpy as np
import Constants as Constants
from PIL import Image
from os.path import join

cascade_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')


# wrapping CSV strings into numpy arrays
def convert_data_to_image(data):
    image = np.fromstring(str(data), dtype=np.uint8, sep=' ').reshape((Constants.FACE_SIZE, Constants.FACE_SIZE))
    image = Image.fromarray(image).convert('RGB')

    # flip the image
    image = np.array(image)[:, :, ::-1].copy()
    image = format_image(image)
    return image


# crop the image to match the network specs.
# separate the faces in the image and get the one with the largest area
def format_image(given_image):
    given_image = cv2.cvtColor(given_image, cv2.COLOR_BGR2GRAY)

    # for image borders
    border = np.zeros((150, 150), np.uint8)
    border[:, :] = 200
    border[
        int((150/2) - (Constants.FACE_SIZE/2)): int((150/2) + (Constants.FACE_SIZE/2)),
        int((150/2) - (Constants.FACE_SIZE/2)): int((150/2) + (Constants.FACE_SIZE/2))
    ] = given_image

    given_image = border
    detected_faces = cascade_classifier.detectMultiScale(given_image, scaleFactor=1.3, minNeighbors=5, minSize=(48, 48),
                                                         flags=cv2.CASCADE_SCALE_IMAGE)

    if detected_faces:
        max_face = detected_faces[0]
        for face in detected_faces:
            if max_face[2] * max_face[3] < face[2] * face[3]:
                max_face = face

        #  cut the face
        face = max_face
        given_image = given_image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]

        # resize the image to fit network specs
        try:
            given_image = cv2.resize(given_image, (Constants.FACE_SIZE, Constants.FACE_SIZE),
                                     interpolation=cv2.INTER_CUBIC) / 255.0
        except Exception:
            print('Exception happened when resizing the image: ' + str(Exception))
            return None
        return given_image
    else:  # no face was detected
        return None


# in the emotion array, everything is set to 0 except for some specific emotion index is set to 1
def encode_emotion_array(index):
    array = np.zeros(len(Constants.EMOTIONS))
    array[index] = 1.0
    return array


# main
data = pd.read_csv(join(Constants.DATA_DIR, Constants.DATASET_CSV_FILENAME))
images = []
labels = []
total = data.shape[0]

print("Started converting...\n")
for i, row in data.iterrows():
    emotion = encode_emotion_array(row['emotion'])
    currImage = convert_data_to_image(row['pixels'])

    if currImage is not None:
        labels.append(emotion)
        images.append(currImage)

    print("Conversion progress: {}/{}".format(i + 1, total))

print("Total number of images: " + str(len(images)))

np.save(join(Constants.DATA_DIR, Constants.DATA_IMAGE_FILE), images)
np.save(join(Constants.DATA_DIR, Constants.DATA_LABEL_FILE), labels)
