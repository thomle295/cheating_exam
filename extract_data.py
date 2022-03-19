import cv2
from HeadPose import HeadPose
from GazeTracking.gaze_tracking import GazeTracking
from FaceDetect.face_detect import FaceDetect

FRAME_PER_TIME = 10


def main():
    webcam = cv2.VideoCapture("../Data/trainning_set/0_2.mp4")

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
    # quit()
    # webcam = cv2.VideoCapture(0)
    gaze = GazeTracking()
    face_detect = FaceDetect()
    head_pose = HeadPose()

    data_10time_10frame = []
    frame_count = 0

    file = open('./CSV/data_10time_10frame_train_0_2.csv', 'w')

    while True:
        frame_count += 1
        # print(frame_count)
        _, frame = webcam.read()

        # face detection
        num_face = face_detect.count_face(frame)

        gaze.refresh(frame)
        frame = gaze.annotated_frame()

        head_pose.refresh(frame)
        frame = head_pose.annotated_frame()

        data = ",".join([
            str(num_face),
            str(gaze.vertical_ratio()),
            str(gaze.horizontal_ratio()),
            str(head_pose.x_face),
            str(head_pose.y_face)
        ])

        if frame_count % FRAME_PER_TIME == 0:
            data_10time_10frame.append(data)

            if len(data_10time_10frame) > 10:
                for row in data_10time_10frame:
                    file.write(row)
                    file.write(",")
                # print(num_face)
                # if (frame_count >= fps*4 and frame_count <= fps*36) or (num_face > 1):
                #     file.write("1")
                # else:
                #     file.write("0")
                file.write("0")
                file.write("\n")
                data_10time_10frame.pop(0)

        # if len(data_10time_10frame) > 10:
        #     for row in data_10time_10frame:
        #         file.write(row)
        #         file.write(",")
        #     file.write("\n")
        #     data_10time_10frame.pop(0)
        # print(
        #     "head:(",
        #     head_pose.x_face,
        #     ",",
        #     head_pose.y_face,
        #     ") ; eye:(",
        #     gaze.horizontal_ratio(),
        #     ", ",
        #     gaze.vertical_ratio(),
        #     ")"
        #     )

        cv2.imshow("Demo", frame)

        if cv2.waitKey(1) == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
