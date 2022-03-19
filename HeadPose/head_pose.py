import cv2
import mediapipe as mp
import numpy as np


class HeadPose(object):
    """
    This class tracks the direction of face.
    """

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.frame = None
        self.x_face = None
        self.y_face = None
        self.nose_2d = None
        self.nose_3d = None
        self.rot_vec = None
        self.trans_vec = None
        self.cam_matrix = None
        self.dist_matrix = None

        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def _analyze(self):
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

        # To improve performance
        frame.flags.writeable = False

        # Get the result
        results = self.face_mesh.process(frame)

        # To improve performance
        frame.flags.writeable = True

        # Convert the color space from RGB to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = frame.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            self.nose_2d = (lm.x * img_w, lm.y * img_h)
                            self.nose_3d = (
                                lm.x * img_w, lm.y * img_h, lm.z * 8000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])

                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                self.cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                            [0, focal_length, img_w / 2],
                                            [0, 0, 1]])

                # The Distance Matrix
                self.dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, self.rot_vec, self.trans_vec = cv2.solvePnP(
                    face_3d, face_2d, self.cam_matrix, self.dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(self.rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                self.x_face = angles[0] * 360
                self.y_face = angles[1] * 360

    def refresh(self, frame):
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame = frame
        self._analyze()

    def annotated_frame(self):
        frame = self.frame.copy()

        # Display the nose direction
        nose_3d_projection, jacobian = cv2.projectPoints(
            self.nose_3d, self.rot_vec,
            self.trans_vec, self.cam_matrix, self.dist_matrix)

        p1 = (int(self.nose_2d[0]), int(self.nose_2d[1]))
        p2 = (int(nose_3d_projection[0][0][0]),
              int(nose_3d_projection[0][0][1]))

        cv2.line(frame, p1, p2, (255, 0, 0), 2)

        return frame
