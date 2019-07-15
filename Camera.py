import cv2
import Constants as Constants
from CNN import NeuralNetworkModel

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')


# modify the image to network requirements and return it
def format_image(given_image):
    if len(given_image.shape) > 2 and given_image.shape[2] == 3:
        given_image = cv2.cvtColor(given_image, cv2.COLOR_BGR2GRAY)
    else:
        given_image = cv2.imdecode(given_image, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    detected_faces = face_cascade.detectMultiScale(given_image, scaleFactor=1.3, minNeighbors=5, minSize=(48, 48),
                                                   flags=cv2.CASCADE_SCALE_IMAGE)

    # if found faces
    if len(detected_faces) > 0:
        max_face = detected_faces[0]
        for face in detected_faces:
            if max_face[2] * max_face[3] < face[2] * face[3]:
                max_face = face
        # crop image to face
        face = max_face
        given_image = given_image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]

        # crop the image to the fit requirements
        try:
            given_image = cv2.resize(given_image, (Constants.FACE_SIZE, Constants.FACE_SIZE),
                                     interpolation=cv2.INTER_CUBIC)/255.0
        except Exception:
            print("Couldn't resize the image")
            return None
        return given_image
    else:
        return None


video_capture = cv2.VideoCapture(0)
neural_network_model = NeuralNetworkModel()
neural_network_model.create_model()
neural_network_model.load_weights('model_weights')

# from OpenCVs documentation
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    result = neural_network_model.predict(format_image(frame))
    print(result)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(48, 48),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if result is not None:
        for index, emotion in enumerate(Constants.EMOTIONS):
            cv2.putText(frame, emotion, (15, index * 20 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.rectangle(frame, (130, index * 20 + 10), (130 +
                                                          int(result[0][index] * 100), (index + 1) * 20 + 4),
                          (255, 0, 0), -1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
video_capture.release()
cv2.destroyAllWindows()
