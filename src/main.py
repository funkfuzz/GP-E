import argparse
import tensorflow as tf
import cv2
import numpy as np
from calibration import camera_params as cp
from utils import utils
from core.Face_Detection import Face_Detection
from core.Landmarks_Detection import Facial_Landmarks
from core.Gaze_Detection import Gaze_Detection
import core.head_pose as hp

parser = argparse.ArgumentParser()
parser.add_argument('--input_camera_id', type=int, choices = range(0,10), default='0', help='Select the input camera device id. Default for webcam is 0.')
parser.add_argument('--gaze_method', choices=['simple', 'complex'], default='simple', help='Choose the method for gaze direction estimation. Options are simple or complex')
parser.add_argument('--auto_calib', type=bool, default=False, help = 'Perform auto calibration of the point of gaze estimation to compensate for different persons head locations')
parser.add_argument('--smooth_samples_gazedir', type=int, choices = range(1,10), default='1', help='Average the gaze direction visualisation over the last n estimations')
parser.add_argument('--smooth_samples_pog', type=int, choices = range(1,10), default='1')



def main():
    args = parser.parse_args()
    device_id = args.input_camera_id
    calibration = args.auto_calib
    gaze_history_max_len = args.smooth_samples_gazedir
    pog_history_max_len = args.smooth_samples_pog

    # some tf config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # do not allow tensorflow to take all of the gpu for itself
    config.log_device_placement = False

    with tf.Session(config=config) as sess:
        # the gaze detection object
        gaze_detector = Gaze_Detection(sess)
        # set the head model
        gaze_detector._3d_head = hp.HEAD_MODEL_3D
        # setting the camera intrinsic mat
        gaze_detector._intr_mat = cp.cam_intrinsic_mat

        video = cv2.VideoCapture(device_id)
        # adjust video_width and video_height according to your input image source
        video.set(3, cp.video_width)
        video.set(4, cp.video_height)

        if not video.isOpened():
            print("No video input. Check connection to camera.")
            exit(-1)

        ret, frame = video.read()
        frame = cv2.flip(frame, 1)

        # initialize objects
        face_detector = Face_Detection(video_width=cp.video_width,video_height=cp.video_height)
        landmark_detector =Facial_Landmarks()
        # init some variables
        frame_index = 0
        fn_max = 1

        while ret:
            # adjust size if necessary
            frame_height, frame_width, _ = frame.shape
            if (frame_width > 1024):
                frame_height = int(1024 * frame_height / frame_width)
                frame_width = 1024
                frame = cv2.resize(frame, (frame_width, frame_height), cv2.INTER_LINEAR)

            k = cv2.waitKey(1) & 0xFF
            if k == ord(' '):
                while True:
                    k = cv2.waitKey(5) & 0xFF
                    if k == ord(' '):
                        break

            # contains the input image
            input_stream = {'image_gray':cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 'image_bgr':frame}  # adjust if needed

            # detect face every 'detection_rate' frames
            if frame_index % face_detector.detection_rate == 0:
                detections = face_detector.face_detector(input_stream['image_gray'], 0)

            if detections:
                face_detector.faces_detected = True
                bbox = face_detector.bounding_box_det(detections, frame)
                # facial landmarks
                facial_landmarks = landmark_detector.get_facial_landmarks(input_stream['image_gray'], bbox)
                # face tracking
                face_detector.track_face(input_stream['image_gray'], bbox)
                # clear
                detections = []

            else:
                face_detector.faces_detected = False

            if face_detector.tracking:
                # two types of bbox output needed for the different functions
                # dlib facial landm est works with dlib.rectangle --> rect
                # bbox is needed for the gaze estimation --> iterable object
                rect, bbox = face_detector.bounding_box_track(facial_landmarks,frame)
                input_stream['faces'] = [{'box':bbox}]

                # estimate facial landmarks again
                # facial_landmarks = landmark_detector.estimate_facial_landmarks(input_stream['image_gray'], rect)
                input_stream['faces'][0]['landmarks'] = facial_landmarks

                # predict head orientation
                euler_angles, rotation_mat, translation_vec, pose_mat = hp.get_head_pose(facial_landmarks)

                # pass orientation params to the gaze estimator
                gaze_detector._rot_matrix = rotation_mat
                gaze_detector._translation_vec = translation_vec

                gaze_detector._pose_mat = pose_mat

                # predict eye heatmaps, landmarks and radiues using GazeML
                gaze_detector.predict(input_stream)

                for face in input_stream['faces']:
                    if ('gaze' not in face):
                        continue

                # for each eye
                for j in range(gaze_detector.batch_size):
                    eye_landmarks, eye_side, can_use_eye, _, _ = gaze_detector._landmarks_from_heatmaps(face, j)
                    eyelid_landmarks = eye_landmarks[0:8,:]
                    iris_landmarks = eye_landmarks[8:16, :]
                    iris_centre = eye_landmarks[16, :]
                    eyeball_centre = eye_landmarks[17, :]
                    eyeball_radius = np.linalg.norm(eye_landmarks[18,:]-eye_landmarks[17,:])

                    # store for visualization
                    gaze_detector.eye_landmarks.append(np.round(eye_landmarks).astype(np.int32))

                    # save gaze history to smooth gaze visualisation output
                    amount_eyes_in_frame = len(face['eyes']) # frame['eyes'] #only len(eyes)
                    if len(gaze_detector.all_gaze_histories) != amount_eyes_in_frame:
                        gaze_detector.all_gaze_histories = [list() for _ in range(amount_eyes_in_frame)]
                    gaze_history = gaze_detector.all_gaze_histories[j]

                    # ToDo: add argparser to switch between the methods
                    # estimate gaze direction
                    # simple
                    #current_gaze = gazeEstimator.simple_gaze_estimation(iris_centre, eyeball_centre, eyeball_radius)
                    # model fitting method
                    current_gaze = gaze_detector.get_gaze_direction_complex(iris_landmarks, iris_centre, eyeball_centre,
                                                                            eyeball_radius)

                    # store
                    gaze_history.append(current_gaze)

                    # calculate gaze from last 'gaze_history_max_len' samples
                    # ToDo smoothen the pog by finding a distribution
                    # ToDo Apply Kalman Filter for gaze tracking when face is ocluded ?
                    if len(gaze_history) > gaze_history_max_len:
                        gaze_history = gaze_history[-gaze_history_max_len:]

                    # ToDo: move to the drawing part further down?
                    # draw gaze
                    if can_use_eye:
                        cv2.drawMarker(
                            input_stream['image_bgr'], tuple(np.round(eyeball_centre).astype(np.int32)),
                            color = (0, 255, 0), markerType = cv2.MARKER_CROSS, markerSize = 4,
                            thickness = 1, line_type = cv2.LINE_AA)
                        # visualize the gaze direction
                        gaze_detector.draw_gaze_direction(input_stream['image_bgr'], eyeball_centre, np.mean(gaze_history, axis=0),
                                                          length=100.0)
                    else:
                        gaze_history.clear()

                    # estimate point of gaze for the current eye
                    gaze_detector.pog[:, j], gaze_detector.pog_i[:, j] = gaze_detector.get_pog(
                        np.mean(gaze_history, axis=0), eye_side)
                    # pog for left and right eye estimated?
                    if not gaze_detector.pog[0:1, :].all():
                        continue

                    else:
                        # pog average between both eyes
                        pog, pog_i = gaze_detector.get_average_pog(
                            euler_angles[1])  # ToDo: implement better averaging method

                        if calibration:
                            continue # ToDo implement a calibration routine


                        # collect all pog
                        gaze_detector._all_pog_history.append(pog)
                        gaze_detector._all_pog_history_i.append(pog_i)

                        # in camera coords
                        pog_history = gaze_detector._all_pog_history
                        # in image coords
                        pog_history_i = gaze_detector._all_pog_history_i
                        if len(gaze_detector._all_pog_history) > pog_history_max_len:
                            pog_history = gaze_detector._all_pog_history[-pog_history_max_len:]
                            pog_history_i = gaze_detector._all_pog_history_i[-pog_history_max_len:]

                        # average pog over the last 'pog_history_max_len' samples
                        # in camera coords
                        pog_mean = np.mean(pog_history, axis=0)
                        # in image coords
                        pog_mean_i = np.mean(pog_history_i, axis=0)

                        # visualize eye landmarks
                        for i in range(gaze_detector.batch_size):
                            for (a, b) in gaze_detector.eye_landmarks[i]:
                                cv2.circle(input_stream['image_bgr'], (a, b), 2, (0, 222, 0), -1)

                        # visualize pog
                        # origin left eye
                        cv2.arrowedLine(input_stream['image_bgr'], tuple(gaze_detector.eye_landmarks[1][17,:]),
                                        tuple(np.round(pog_mean_i).astype(np.int32)), (255, 0, 11),
                                        1, cv2.LINE_AA, tipLength = 0.01)
                        # origin right eye
                        cv2.arrowedLine(input_stream['image_bgr'], tuple(gaze_detector.eye_landmarks[0][17,:]),
                                        tuple(np.round(pog_mean_i).astype(np.int32)), (255, 0, 11),
                                        1, cv2.LINE_AA, tipLength = 0.01)

                        # clear the estimations
                        gaze_detector.pog[:, :] = 0
                        gaze_detector.pog_i[:, :] = 0
                        gaze_detector.eye_landmarks.clear()

                # draw landmarks
                for (a, b) in facial_landmarks:
                    cv2.circle(input_stream['image_bgr'], (a, b), 2, (0, 128, 255), -1)

                origin = (facial_landmarks[21] + facial_landmarks[22])/2
                # head orientation
                utils.draw_axis(input_stream['image_bgr'], euler_angles[1], - euler_angles[0], euler_angles[2], origin[0],
                                origin[1]*0.9, size=100)  # use when hnet
                # bounding box
                cv2.rectangle(input_stream['image_bgr'], (bbox[0],bbox[1]),(bbox[2],bbox[3]), (0, 128, 255), 1)

            cv2.imshow("Visualisation", input_stream['image_bgr'])

            if ((cv2.waitKey(10) & 0xFF) == ord('q')):
                break

            # stream forever
            fn = 0
            while (fn < fn_max):
                fn+=1
                ret, frame = video.read()

        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()




