import cv2
from HeadPose import HeadPose
from GazeTracking.gaze_tracking import GazeTracking
from FaceDetect.face_detect import FaceDetect
from sklearn.preprocessing import PolynomialFeatures
import pickle
import sys

FRAME_PER_TIME = 10


def predict():
    # print(sys.argv)
    webcam = cv2.VideoCapture(sys.argv[1])
    file = open(sys.argv[2], 'w')

    # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    fps = webcam.get(cv2.CAP_PROP_FPS)
    frame_count = int(webcam.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps

    print('fps = ' + str(fps))
    print('number of frames = ' + str(frame_count))
    print('duration (S) = ' + str(duration))
    minutes = int(duration/60)
    seconds = duration % 60
    print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))

    gaze = GazeTracking()
    face_detect = FaceDetect()
    head_pose = HeadPose()
    poly_reg = PolynomialFeatures(degree=4)
    loaded_model = pickle.load(
        open("./Model/finalized_logistist_model.sav", 'rb'))
    property = 0
    input = []

    while True:
        frame_count += 1
        # print(frame_count)
        _, frame = webcam.read()

        # face detection
        num_face_mediapipe_method = face_detect.count_face_mediapipe_method(
            frame)
        # num_face_haarcascade = face_detect.count_face_haarcascade(frame)
        # print(num_face_mediapipe_method)

        if num_face_mediapipe_method > 0:
            gaze.refresh(frame)
            frame = gaze.annotated_frame()

            head_pose.refresh(frame)
            frame = head_pose.annotated_frame()
            data = [
                num_face_mediapipe_method,
                gaze.vertical_ratio(),
                gaze.horizontal_ratio(),
                head_pose.x_face,
                head_pose.y_face
            ]
        else:
            data = [
                0,
                0,
                0,
                0,
                0
            ]

        if frame_count % FRAME_PER_TIME == 0:
            input = input + data

            if len(input) >= 55:
                X = poly_reg.fit_transform([input])
                property = loaded_model.predict_proba(X)[0][1]
                file.write(str(frame_count/fps))
                file.write(",")
                file.write(str(property))
                file.write("\n")
                input = input[5:]

        property = property
        # print(property)
        cv2.putText(frame, "Cheating rate: " + str(property),
                    (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

        cv2.imshow("Demo", frame)

        if cv2.waitKey(1) == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    predict()
