import math
import numpy as np
import cv2
from .pupil import Pupil


class Eye(object):
    """
    This class creates a new frame to isolate the eye and
    initiates the pupil detection.
    """

    LEFT_EYE_POINTS = [466, 388, 387, 386, 385, 384, 398,
                       263, 249, 390, 373, 374, 380, 381, 382, 362]
    RIGHT_EYE_POINTS = [246, 161, 160, 159, 158, 157,
                        173, 33, 7, 163, 144, 145, 153, 154, 155, 133]

    def __init__(self, original_frame, landmarks, side, calibration):
        self.frame = None
        self.origin = None
        self.center = None
        self.pupil = None
        self.landmark_points = None

        self._analyze(original_frame, landmarks, side, calibration)

    @staticmethod
    def _middle_point(p1_x, p1_y, p2_x, p2_y):
        """Returns the middle point (x,y) between two points

        Arguments:
            p1 (dlib.point): First point
            p2 (dlib.point): Second point
        """
        x = int((p1_x + p2_x) / 2)
        y = int((p1_y + p2_y) / 2)
        return (x, y)

    def _isolate(self, frame, landmarks, points):
        """Isolate an eye, to have a frame without other part of the face.

        Arguments:
            frame (numpy.ndarray): Frame containing the face
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            points (list): Points of an eye (from the 68 Multi-PIE landmarks)
        """
        # region = np.array(
        #     [(landmarks.part(point).x, landmarks.part(point).y) for point in points])
        # region = region.astype(np.int32)
        faceLms = landmarks.multi_face_landmarks[0]
        ih, iw, ic = frame.shape
        region = np.array(
            [(faceLms.landmark[point].x*iw, faceLms.landmark[point].y*ih)
             for point in points]
        )
        region = region.astype(np.int32)
        self.landmark_points = region

        # Applying a mask to get only the eye
        height, width = frame.shape[:2]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.equalizeHist(frame)
        black_frame = np.zeros((height, width), np.uint8)
        mask = np.full((height, width), 255, np.uint8)
        cv2.fillPoly(mask, [region], (0, 0, 0))
        eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)
        # cv2.imwrite("../bitwise_not.jpg", frame)

        # Cropping on the eye
        margin = 5
        # print(region[:, 0])
        min_x = np.min(region[:, 0]) - margin
        max_x = np.max(region[:, 0]) + margin
        min_y = np.min(region[:, 1]) - margin
        max_y = np.max(region[:, 1]) + margin
        # print(min_x, max_x, min_y, max_y)
        self.frame = eye[min_y:max_y, min_x:max_x]
        self.origin = (min_x, min_y)
        # cv2.imwrite("../eye_cut.jpg", self.frame)
        # quit()
        height, width = self.frame.shape[:2]
        self.center = (width / 2, height / 2)

    def _blinking_ratio(self, original_frame, landmarks, points):
        """Calculates a ratio that can indicate whether an eye is closed or not.
        It's the division of the width of the eye, by its height.

        Arguments:
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            points (list): Points of an eye (from the 68 Multi-PIE landmarks)

        Returns:
            The computed ratio
        """
        faceLms = landmarks.multi_face_landmarks[0]
        ih, iw, ic = original_frame.shape
        left_point = (
            int(faceLms.landmark[points[7]].x*iw),
            int(faceLms.landmark[points[7]].y*ih)
            )
        right_point = (
            int(faceLms.landmark[points[15]].x*iw),
            int(faceLms.landmark[points[15]].y*ih)
        )
        top_point = self._middle_point(
            int(faceLms.landmark[points[3]].x*iw),
            int(faceLms.landmark[points[3]].y*ih),
            int(faceLms.landmark[points[4]].x*iw),
            int(faceLms.landmark[points[4]].y*ih)
        )

        bottom_point = self._middle_point(
            int(faceLms.landmark[points[11]].x*iw),
            int(faceLms.landmark[points[11]].y*ih),
            int(faceLms.landmark[points[12]].x*iw),
            int(faceLms.landmark[points[12]].y*ih)
        )

        eye_width = math.hypot(
            (left_point[0] - right_point[0]),
            (left_point[1] - right_point[1])
            )

        eye_height = math.hypot(
            (top_point[0] - bottom_point[0]),
            (top_point[1] - bottom_point[1])
            )
        try:
            ratio = eye_width / eye_height
        except ZeroDivisionError:
            ratio = None

        return ratio

    def _analyze(self, original_frame, landmarks, side, calibration):
        """Detects and isolates the eye in a new frame, sends data to the calibration
        and initializes Pupil object.

        Arguments:
            original_frame (numpy.ndarray): Frame passed by the user
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            side: Indicates whether it's the left eye (0) or the right eye (1)
            calibration (calibration.Calibration): Manages the binarization threshold value
        """
        if side == 0:
            points = self.LEFT_EYE_POINTS
        elif side == 1:
            points = self.RIGHT_EYE_POINTS
        else:
            return

        self.blinking = self._blinking_ratio(original_frame, landmarks, points)
        self._isolate(original_frame, landmarks, points)

        if not calibration.is_complete():
            calibration.evaluate(self.frame, side)

        threshold = calibration.threshold(side)
        self.pupil = Pupil(self.frame, threshold)
