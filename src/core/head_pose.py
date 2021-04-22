import cv2
import numpy as np
from src.calibration import camera_params as cp

# 3D head model to use for solvePnP()
HEAD_MODEL_3D = np.float32([[-70, 5, 97],           #0  landmark from the 68 dlib facial landmarks predictor
                            [-58, 72, 97],          #4
                            [0, 119, 0],            #8
                            [58, 72, 97],           #12
                            [70, 5, 97],            #16
                            [-55, -10, 20],         #17
                            [55, -10, 20],          #26
                            [0, 0, 0],              #O
                            [0, 48, -20],           #27
                            [0, 50, -5],            #33
                            [-34, 5, 17.0],         #36
                            [-14.0, 5, 14],         #39
                            [14, 5, 14],            #42
                            [34, 5, 17],            #45
                            [0, 72, -10],           #62
                            [-25, 6, 24],           #eyeball right
                            [25, 6, 24]])           #eyeball left



def get_head_pose(flandmarks):
    ''' Estimates the 3d orientation of the head.

        Estimates the head pose by utilizing cv2.solvePnP().

    :param flandmarks: 2D facial landmarks
    :return: the euler angles of rotation of the persons head,
             a rotation matrix,
             translation vector,
             pose matrix
    '''

    # head model from 2d facial landmarks
    head_model_2d  = np.float32([flandmarks[0], flandmarks[4], flandmarks[6], flandmarks[12], flandmarks[16],
                                 flandmarks[17], flandmarks[26], flandmarks[27], flandmarks[30], flandmarks[33],
                                 flandmarks[36], flandmarks[39], flandmarks[42], flandmarks[45], flandmarks[62]])

    # in camera coords
    _, rotation_vec, translation_vec  = cv2.solvePnP(HEAD_MODEL_3D[0:15], head_model_2d, cp.cam_intrinsic_mat, cp.dist_coeffs)

    # transform
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))

    # transform to euler
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    return euler_angles, rotation_mat, translation_vec, pose_mat