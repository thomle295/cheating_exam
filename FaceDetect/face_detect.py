import mediapipe as mp
import cv2


class FaceDetect(object):

    def __init__(self):
        mpFaceMesh = mp.solutions.face_mesh
        cascPath = "FaceDetect/haarcascade_frontalface_default.xml"
        self.faceCascade = cv2.CascadeClassifier(cascPath)
        self.faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)

    def count_face_mediapipe_method(self, frame):
        results = self.faceMesh.process(frame)
        if results.multi_face_landmarks:
            return len(results.multi_face_landmarks)
        return 0

    def count_face_haarcascade(self, frame):
        # Read the image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        return len(faces)
