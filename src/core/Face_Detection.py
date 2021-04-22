import dlib
import numpy as np




class Face_Detection:
    """ Class for Face detection and tracking.

        This class implements methods that enable face detection and tracking to be performed.

    """
    def __init__(self, video_width = 640, video_height = 480 ):
        self._video_width = video_width
        self._video_heigh = video_height
        self._best_bounding_box_width = video_width
        self._best_bounding_box_height = video_height
        self.face_detector = dlib.get_frontal_face_detector()
        # perform face detection every n frames
        self.detection_rate = 20
        self.faces_detected = False
        self.tracker = dlib.correlation_tracker()
        self.tracking = False
        # scale parameters
        self.SCALE_H = 0.15
        self.BBOX_OFFSET = 10

    def get_eye_center(self, facial_landmarks):
        ''' Estimates the eye centers from the facial landmarks.

            Currently supports only 68 facial landmarks model by dlib.

        :param facial_landmarks: the 68 facial landmarks estimated by shape_predictor_68_face_landmarks.dat
        :return: eye_center_left, eye_center_right: coordinates of the eye centers
        '''

        outer_corner_eye_l = facial_landmarks[36]
        inner_corner_eye_l = facial_landmarks[39]
        outer_corner_eye_r = facial_landmarks[45]
        inner_corner_eye_r = facial_landmarks[42]

        # linear regression
        x = [outer_corner_eye_l[0], inner_corner_eye_l[0], outer_corner_eye_r[0], inner_corner_eye_r[0]]
        y = [outer_corner_eye_l[1], inner_corner_eye_l[1], outer_corner_eye_r[1], inner_corner_eye_r[1]]
        A = np.vstack([x, np.ones(len(x))]).T
        k, b = np.linalg.lstsq(A, y, rcond=None)[0]

        x_left = (outer_corner_eye_l[0] + inner_corner_eye_l[0]) / 2
        x_right = (outer_corner_eye_r[0] + inner_corner_eye_r[0]) / 2
        eye_center_left = np.array([np.int32(x_left), np.int32(x_left * k + b)])
        eye_center_right = np.array([np.int32(x_right), np.int32(x_right * k + b)])

        return eye_center_left, eye_center_right



    def bounding_box_det(self, detections, input_frame):
        ''' Estimates a bounding box around the biggest face in the image
            when no tracking is performed.

        :param detections: detected face
        :param input_frame: the current image frame to process.
        :return: a rectangle bounding box around the face roi.
        '''

        x_min_ = [0] * len(detections)
        y_min_ = [0] * len(detections)
        x_max_ = [0] * len(detections)
        y_max_ = [0] * len(detections)

        bounding_box_w = [0] * len(detections)
        bounding_box_h = [0] * len(detections)

        for index, det in enumerate(detections):
            x_min_[index] = det.left()
            y_min_[index] = det.top()
            x_max_[index] = det.right()
            y_max_[index] = det.bottom()
            bounding_box_w[index] = abs(x_max_[index] - x_min_[index])
            bounding_box_h[index] = abs(y_max_[index] - y_min_[index])

        idx_closest_face = bounding_box_w.index(max(bounding_box_w))

        # new bounding box must not be smaller than the half of the old one
        if bounding_box_w[idx_closest_face] < self._best_bounding_box_width and bounding_box_w[idx_closest_face]/self._best_bounding_box_width > 0.5 \
            or bounding_box_h[idx_closest_face] < self._best_bounding_box_height and bounding_box_h[idx_closest_face]/self._best_bounding_box_height:
            self._best_bounding_box_width = bounding_box_w[idx_closest_face]
            self._best_bounding_box_height = bounding_box_h[idx_closest_face]

        center = detections[idx_closest_face].center()
        x_min = center.x - self._best_bounding_box_width / 2
        y_min = center.y - self._best_bounding_box_height / 2
        x_max = center.x + self._best_bounding_box_width / 2
        y_max = center.y + self._best_bounding_box_height / 2

        # bounding box within the image
        x_min = max(x_min,0)
        y_min = max(y_min,0)
        x_max = min(input_frame.shape[1], x_max)
        y_max = min(input_frame.shape[0], y_max)

        # initialize RoI
        rect = dlib.rectangle(int(x_min), int(y_min), int(x_max), int(y_max))

        return rect

    def bounding_box_track(self, f_landmarks, input_frame):
        ''' Estimates a bounding box around the biggest face in the image
            when a face was already detected and facial landmarks are known.

        :param detections: detected face
        :param input_frame: the current image frame to process.
        :return: a rectangle bounding box around the face roi.
        '''

        left_eye_center, right_eye_center = self.get_eye_center(f_landmarks)  # FD

        eyes_center = ((left_eye_center[0] + right_eye_center[0]) * 0.5,
                       (left_eye_center[1] + right_eye_center[1]) * 0.5)

        offset = int(self._best_bounding_box_width * self.SCALE_H)

        # size of bbox
        x_min = eyes_center[0] - self._best_bounding_box_width/2 - self.BBOX_OFFSET
        y_min = eyes_center[1] - self._best_bounding_box_height/2 - self.BBOX_OFFSET + offset
        x_max = eyes_center[0] + self._best_bounding_box_width/2 + self.BBOX_OFFSET
        y_max = eyes_center[1] + self._best_bounding_box_height/2 + self.BBOX_OFFSET + offset

        # bbox for face detection
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(input_frame.shape[1], x_max)
        y_max = min(input_frame.shape[0], y_max)

        rect = dlib.rectangle(int(x_min), int(y_min), int(x_max), int(y_max))
        box = (int(x_min), int(y_min), int(x_max), int(y_max))  # iterable object

        return rect, box

    def track_face(self, input_frame, roi):
        ''' Tracks the detected face.

            Tracks the detected face, so that no heavy face detection has to be performed each frame.
            Helps speed up estimation.

        :param input_frame: the current input image
        :param roi: the region of interest to be tracked. In this case the face
        '''

        self.tracker.start_track(input_frame, roi)
        self.tracking = True
