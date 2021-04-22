import os
import dlib
import numpy as np

class Facial_Landmarks:
    """ The class for facial landmarks detection.

    """
    def __init__(self, predictor_path = os.path.dirname(os.path.realpath(__file__)) + '/../models/shape_predictor_68_face_landmarks.dat'):
        self.shape_predictor = dlib.shape_predictor(predictor_path)

    def get_facial_landmarks(self, input_frame, face_roi):
        landmarks = self.shape_predictor(input_frame, face_roi)
        # create a list of (x,y) - coordinates
        landmark_coords = np.zeros((68, 2), dtype=int)

        # loop over the landmarks and convert them to x,y - coords
        for i in range(0, 68):
            landmark_coords[i] = (landmarks.part(i).x, landmarks.part(i).y)

        return landmark_coords