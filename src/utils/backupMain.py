import os
import tensorflow as tf
import torch
from torch.autograd import Variable
import torchvision
from torchvision import transforms
import torch.backends.cudnn as cudnn
import cv2
import dlib
import numpy as np
from scipy.spatial.transform import Rotation as R
# ToDo: rename this
import torch.nn.functional as F
#ToDo: rename this
from PIL import Image

from utils import utils
from models import hopenet # ToDo: implement these
# from zeph.landmarksEstimatorimport LandmarksEstimator
from core.Gaze_Detection import Gaze_Detection


# values from Zephanja - note what they are.
# ToDo: add meaningful names
h_ = 0.15
r = 1
q = 20 # perform face detection every q frames
c = 6

d= 0.233 #ToDo: what is this?
frame_num = 0 # ToDo: find a better place in Code to initialize this

# HP g5 Webcam camera matrix
# ToDo: adjust for the new camera
video_width = 640
video_height = 480
# webcam data from calibration with chAruco
# scaled focal length
f_x = 487
f_y = 653

cam_intrinsic_mat = np.array(
    [[f_x, 0, video_width/2],
     [0, f_y, video_height/2],
     [0, 0, 1]], dtype = "double"
) # ToDo: dynamic type double necessary?

# distortion coeffs Webcam
dist_coeffs = np.array([-0.2, 0.2, 0.0, 0.0, -0.1])

# 3D landmarks in world cooordinate system
object_pts = np.float32([[-70.25, 5.0, 96.85],    #0
                         [-58.3, 72.0, 96.85],    #4
                         [0.0, 119.3, 0],         #8
                         [58.3, 72.0, 96.85],     #12
                         [70.25, 5.0, 96.85],     #16
                         [-55.45, -10.0, 20.0],   #17
                         [55.45, -10.0, 20.0],    #26
                         [0, 0, 0],               #O
                         [0, 48.0, -20.6],        #27
                         [0, 50.6, -5.0],         #33
                         [-34, 5.0, 17.0],        #36
                         [-14.0, 5.0, 14.6],      #39
                         [14.0, 5.0, 14.6],       #42
                         [34.0, 5.0, 17.0],       #45
                         [0.0, 72.0, -10.0],      #62
                         [-24.0, 5.0, 23.0],      #right eyeball
                         [24.0, 5.0, 23.0]])      #left eyeball


# ToDo: change names
# ToDo: maybe store in a separate file utils
def get_head_pose(landmarks):
    ''' The function is able to estimate the head pose.

        Estimates the head pose by utilizing cv2.solvePnP.

    :param landmarks: 2D facial landmarks
    :return: the euler angles of rotation of the persons head,
             a rotation matrix,
             translation vector,
             pose matrix
    '''

    # image points from the geneeric head model
    image_pts = np.float323([landmarks[0], landmarks[4], landmarks[6], landmarks[12], landmarks[16],
                             landmarks[17], landmarks[26], landmarks[27], landmarks[30], landmarks[33],
                             landmarks[36], landmarks[39], landmarks[42], landmarks[45], landmarks[62]])

    _, rotation_vec, translation_vec  = cv2.solvePnP(object_pts[0:15], image_pts, cam_intrinsic_mat, dist_coeffs)

    # transform vec to mat
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    # head pose mat in the camera frame
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    # find euler angles
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return euler_angle, rotation_mat, translation_vec, pose_mat

euler_angle = [0,0,0] # ToDo: find better place in code

# ToDo: maybe store in a separate file utils
def get_eye_center(landmarks):
    """ Estimates the centers of the eyes.

    :param landmarks: 2D facial landmarks
    :return: eye center coordinates
    """
    # linear regression
    left_eye_outer = landmarks[36]
    left_eye_inner = landmarks[39]
    right_eye_outer = landmarks[45]
    right_eye_inner = landmarks[42]

    x = [left_eye_outer[0], left_eye_inner[0], right_eye_outer[0], right_eye_inner[0]]
    y = [left_eye_outer[1], left_eye_inner[1], right_eye_outer[1], right_eye_inner[1]]
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]

    x_left = (left_eye_outer[0]+left_eye_inner[0])/2
    x_right = (right_eye_outer[0]+right_eye_inner[0])/2
    left_eye_center = np.aray([np.int32(x_left), np.int32(x_left*k + b)])
    right_eye_center = np.aray([np.int32(x_right), np.int32(x_right*k + b)])

    return left_eye_center, right_eye_center


cudnn.enabled = True
# batch_size = 1
gpu = 0
#ToDo: change path
snapshot_path = os.path_dirname(os.path.realpath(__file__)) + 'pre-trained_models/head_pose_estimation/hopenet_robust..'
print(snapshot_path)
n_frames = 3000

# initialize model
model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

# Dlib face detection model
face_detector = dlib.get_frontal_face_detector()
tracker = dlib.correlation_tracker()

# path to Dlib facial landm detector
#ToDo: change path
#ToDo: change name to s_predictor
shape_predictor_path = os.path.dirname(os.path.realpath(__file__)) + 'models/shape_predictor_68_face_landmarks.dat'
shape_predictor = dlib.shape_predictor(shape_predictor_path)


print("Loading snapshot")
# Load snapshot
saved_state_dict = torch.load(snapshot_path)
model.load_state_dict(saved_state_dict)

# ToDo: move this section somewhere else
transformations = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# ToDo: move this section somewhere else
# enable cuda gpu usage
model.cuda(gpu)
# set to eval mode
model.eval()
# ToDo: move this section somewhere else
total = 0

idx_tensor = [idx for idx in range(66)]
idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

# ToDo: move this section somewhere else
# input sample rate
filter = 1

landmarksEstimation = LandmarksEstimator(shape_predictor_path, video_width, video height)

#ToDo: find out what are these and rename them
flag = 0
scale = 0
center_mean = [0, 0]
center4 = [0, 0]
center3 = [0, 0]
center2 = [0, 0]
center1 = [0, 0]

# initialize
tracker_running = False
best_bbox_width = video_width
best_bbox_height = video_height

# without this configuration tensorflow allocates most ofthe gpu for
# itself (GazeML) and no space is left for Hopenet
config = tf.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
config.log_device_placement = False

with tf.Session(config=config) as sess:
    # create the gaze estimation object
    gazeEstimator = Gaze_Detection(sess)
    # set the head model
    gazeEstimator._3d_head = object_pts
    # setting the camera intrinsic mat  #ToDo change name from _K to _intr_mat
    gazeEstimator._intr_mat = cam_intrinsic_mat
    # ToDo: implement video Input, argparser to switch between web and video
    video = cv2.VideoCapture(0)
    video.set(3, video_width)
    video.set(4, video_height)

    if not video.isOpened():
        print("Unable to connect to camera.")
        exit(-1)

    ret, frame = video.read()
    frame = cv2.flip(frame, 1)

    while ret:
        # check the size and adjust if necessary
        h, w, _ = frame.shape
        if (w > 1024):
            h = int(1024*h/w)
            w = 1024
            frame = cv2.resize(frame, (w, h), cv2.INTER_LINEAR)

        k = cv2.waitKey(1) & 0xFF
        if k == ord(' '):
            while True:
                k = cv2.waitKey(5) & 0xFF
                if k == ord(' '):
                    break

        face_found = 0
        # dict with the input frame
        # three and single channel
        context = {'frame_bgr':cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR), 'gray':frame}  # adjust if needed

        # detect face every q frames
        if frame_num % q == 0:
            dets = face_detector(context['gray'], 0)

        # if face detected find bounding box
        if dets:
            x_min_ = [0]*len(dets)
            y_min_ = [0] * len(dets)
            x_max_ = [0] * len(dets)
            y_max_ = [0] * len(dets)

            bbox_width_ = [0]*len(dets)
            bbox_height_ = [0]*len(dets)

            for index, det in enumerate(dets):
                x_min_[index] = det.left()
                y_min_[index] = det.top()
                x_max_[index] = det.right()
                y_max_[index] = det.bottom()
                bbox_width_[index] - abs(x_max_[index] - x_min_[index])
                bbox_height_[index] - abs(y_max_[index] - y_min_[index])

            idx_biggest_face = bbox_width_.index(max(bbox_width_))
            # new bounding box must not be smaller than the half of the old one
            if bbox_width_[idx_biggest_face] < best_bbox_width and bbox_width_[idx_biggest_face]/best_bbox_width > 0.5 \
                or bbox_height_[idx_biggest_face] < best_bbox_height and bbox_height_[idx_biggest_face]/best_bbox_height:
                best_bbox_width = bbox_width_[idx_biggest_face]
                best_bbox_height = bbox_height_[idx_biggest_face]

            center = dets[idx_biggest_face]
            x_min = center.x - best_bbox_width / 2
            y_min = center.y - best_bbox_height / 2
            x_max = center.x + best_bbox_width / 2
            y_max = center.y + best_bbox_height / 2

            # bounding box within the image
            x_min = max(x_min,0)
            y_min = max(y_min,0)
            x_max = min(frame.shape[1], x_max)
            y_max = min(frame.shape[0], y_max)

            # initialize RoI
            rect = dlib.rectangle(int(x_min), int(y_min), int(x_max), int(y_max))
            # facial landmarks
            #landmarks = landmarksEstimation.detectLandmarks(context['gray'], rect)  # ToDo: no need of an extra class ey?
            landmarks = shape_predictor(context['grey', rect])
            face_found = 1

            # face tracking
            tracker.start_track(context['gray'], rect)
            tracker_running = True

            dets = []

        else:
            face_found = 0

        if tracker_running:
            # find a bounding box around the ace
            left_eye_center, right_eye_center = get_eye_center(landmarks)

            eyes_center = ((left_eye_center[0] + right_eye_center[0])*0.5,
                           (left_eye_center[1] + right_eye_center[1])*0.5)

            offset = int(best_bbox_width*h_)    #ToDo change names of all ambiguous variables c, d , h_
            x_min  = eyes_center[0] - best_bbox_width/2 - c
            y_min = eyes_center[1] - best_bbox_height / 2 - c + offset
            x_max  = eyes_center[0] + best_bbox_width/2 + c
            y_max = eyes_center[1] + best_bbox_height / 2 + c + offset

            # ToDo: figure out why there are x_min2 and find out if it can be removed
            x_min2  =eyes_center[0] - best_bbox_width/2 - d*best_bbox_width
            y_min2 = eyes_center[1] - best_bbox_height / 2 - d*best_bbox_height + offset
            x_max2  =eyes_center[0] + best_bbox_width/2 + d*best_bbox_width
            y_max2 = eyes_center[1] + best_bbox_height / 2 + d*best_bbox_height + offset

            bbox_width = abs(x_max - x_min)
            bbox_height  = abs(y_max - y_min)

            #bbox for face detection
            x_min = max(x_min,0)
            y_min = max(y_min,0)
            x_max = min(frame.shape[1], x_max)
            y_max = min(frame.shape[0], y_max)

            # bbox2 for hnet input
            x_min2 = max(x_min2,0)
            y_min2 = max(y_min2,0)
            x_max2 = min(frame.shape[1], x_max2)
            y_max2 = min(frame.shape[0], y_max2)

            rect = dlib.rectangle(int(x_min), int(y_min), int(x_max), int(y_max))
            box = (int(x_min), int(y_min), int(x_max), int(y_max))   # ToDo: isnt this ambiguous ? why are there 2 rects, cant I use just one
            context['faces'] = [{'box':box}]

            # input image for hnet
            # red bbox in the preview image
            img = context['frame_bgr'][int(y_min2):int(y_max2), int(x_min2):int(x_max2)]

            # convert format
            img = Image.fromarray(img)

            # Transform
            img = transformations(img)
            img_shape = img.size()
            img =img.view(1, img_shape[0], img_shape[1], img_shape[2])
            img = Variable(img).cuda(gpu)

            # predict head pose with hnet
            yaw, pitch, roll = model(img)
            # ToDo rename this F
            yaw_predicted = F.softmax(yaw, dir=1)  # ToDo not sure if its dir
            pitch_predicted = F.softmax(pitch, dir=1)
            roll_predicted = F.softmax(roll, dir=1)

            # continous predictions
            # in degrees
            yaw_predicted = torch.sum(yaw_predicted.data[0]*idx_tensor)*3 - 99
            pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
            roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

            # angles in radians
            # ToDo: not sure if needed, or if not the same as original yaw predicted values - decide if to leave or not
            yaw_predicted_rad= (yaw_predicted*np.pi)/180
            pitch_predicted_rad = (pitch_predicted * np.pi) / 180
            roll_predicted_rad = (roll_predicted * np.pi) / 180

            # adjust the angles to match the output of get_head_pose()
            # get_head_pose() orientation = camera coordinate frame
            theta = np.float32([-pitch_predicted_rad, yaw_predicted_rad, roll_predicted_rad])

            # rotation matrix from hnet euler angles
            rot_mat_hnet_angles = R.from_euler('xyz', theta, degrees=False)  #ToDo rename R to smth more clear
            rot_mat_hnet = rot_mat_hnet_angles.asmatrix()

            # estimate facial landmarks
            #landmarks = landmarksEstimation.detectLandmarks(context['gray'],rect)
            landmarks = shape_predictor(context['grey', rect])
            last_angle = euler_angle[2] # ToDo: not sure if its index 2
            context['faces'][0]['landmarks'] = landmarks

            # predict head pose with get_head_pose()
            euler_angle, rotation_mat, translation_vec, pose_mat = get_head_pose(landmarks)

            # for comparison of hnet and get_head_pose() outs
            # print('Pitch_euler: %s'%(euler_angle[0]))
            # print('Pitch_hope: %s'%(pitch_predicted))
            # print('Yaw_euler: %s'%(euler_angle[1]))
            # print('Yaw_hope: %s'%(yaw_predicted))
            # print('Roll_euler: %s'%(euler_angle[2]))
            # print('Roll_hope: %s'%(roll_predicted))


            # pass orientation params to the gaze estimator
            gazeEstimator._rot_matrix = rotation_mat
            gazeEstimator._translation_vec = translation_vec
            # pose mat as a combo from Hnet and get_head_pose()
            # pose_mat_fusion = np.column_stack([rot_mat_hnet, translation_vec])  ToDo: remove this eventually
            pose_mat_fusion = np.column_stack([rot_mat_hnet, translation_vec])

            # depending on the chosen config
            if solvePnP:   #ToDo not sure if this param is currently defined
                gazeEstimator._pose_mat = pose_mat
            else:
                # pose mat as a combo from Hnet and get_head_pose()
                pose_mat_fusion = np.column_stack([rot_mat_hnet, translation_vec])
                gazeEstimator._pose_mat = pose_mat_fusion

            # predict eye heatmaps, landmarks and radiues using GazeML
            gazeEstimator.predict(context)

            for face in context['faces']:
                if ('gaze' not in face):
                    continue

            # for each eye
            for j in range(gazeEstimator.batch_size):     # ToDo check what the input should be and adjust, Also adjust the name so it is not a private method
                eye_landmarks,eye_side, can_use_eye, can_use_eyelid, can_use_iris = gazeEstimator._landmarks_from_heatmaps(face, j)
                # manually overwrite, depending on the need #ToDo: maybe implement as an input argument, flag ?
                can_use_eye, can_use_eyelid, can_use_iris = True, True, True  #np.asarray(eye_landmarks)
                eyelid_landmarks = eye_landmarks[0:8,:]
                iris_landmarks = eye_landmarks[8:16, :]
                iris_centre = eye_landmarks[16, :]
                eyeball_centre = eye_landmarks[17, :]
                eyeball_radius = np.linalg.norm(eye_landmarks[18,:]-eye_landmarks[17,:])

                # save gazer history to smooth gaze visualisation output
                num_total_eyes_in_frame = len(face['eyes']) # frame['eyes'] #only len(eyes)
                if len(gazeEstimator.all_gaze_histories) != num_total_eyes_in_frame:
                    gazeEstimator.all_gaze_histories = [list() for _ in range(num_total_eyes_in_frame)]
                gaze_history = gazeEstimator.all_gaze_histories[j]

                # estimate gaze direction
                # naive
                current_gaze = gazeEstimator.get_gaze_direction_simple(iris_centre, eyeball_centre, eyeball_radius)
                # model fitting method
                # current_gaze = gazeEstimator.estimate_gaze_from_landmarks(iris_landmarks, iris_centre, eyeball_centre, eyeball_radius)


                # store current gaze
                gaze_history.append(current_gaze)

                # calculate gazze from last 'gaze_history_max_len' samples
                # ToDo smoothen the pog by finding a distribution
                # ToDo Apply Kalman Filter for gaze tracking when face is ocluded ?
                gaze_history_max_len = 1
                if len(gaze_history) > gaze_history_max_len:
                    gaze_history = gaze_history[-gaze_history_max_len:]

                # draw gaze
                if can_use_eye:
                    cv2.drawMarker(
                        context['frame_bgr'], tuple(np.round(eyeball_centre).astype(np.int32)),
                        color = (0, 255, 0), markerType = cv2.MARKER_CROSS, markerSize = 4,
                        thickness = 1, line_type = cv2.LINE_AA)
                    #visualize the gaze direction
                    gazeEstimator.draw_gaze_direction(context['frame_bgr'], eyeball_centre, np.mean(gaze_history, axis=0),
                                                      length=100.0, thickness = 1)
                else:
                    gaze_history.clear()

                # estimate point of gaze for the current eze
                gazeEstimator.pog[:, j], gazeEstimator.pog_i[:, j] = gazeEstimator.get_pog(
                    np.mean(gaze_history, axis=0), eye_side)
                # point of gaze of both eyes known?
                if not gazeEstimator.pog[0:1, :].all():
                    continue

                else:
                    # find the average point of gaze of both eyes ToDo: try using another averaging method
                    PoG_average, PoG_average_i = gazeEstimator.get_average_pog(yaw_predicted)

                    if calibration: # ToDo I think calibraton is not initialized, eventually passable as an argument
                        PoG_average = copy.copy(PoG_average)
                        PoG_average_ = np.append(PoG_average[:2],1)
                        # perform calib using the two dif methods  ToDo: choose one and remove
                        PoG_average_calib = calib_mat1.dot(PoG_average_.T)
                        PoG_average_calib = calib_mat2.dot(PoG_average_.T)

                    # storePoG estimations for smoothing
                    gazeEstimator._all_pog_history.append(PoG_average)
                    gazeEstimator._all_pog_history_i.append(PoG_average_i)

                    # apply some transformations
                    PoG_left_data = (copy.copy(gazeEstimator.pog[:, 0])).tolist()
                    PoG_right_data = (copy.copy(gazeEstimator.pog[:, 1])).tolist()
                    PoG_average_data = (copy.copy(PoG_average)).tolist()
                    gaze_angles_left_data = (gazeEstimator.all_gaze_histories[0][-1]).tolist()
                    gaze_angles_riht_data = (gazeEstimator.all_gaze_histories[1][-1]).tolist()
                    theta_data = (copy.copy(theta)).tolist()
                    data_dict['PoG_left'].append(PoG_left_data)
                    data_dict['PoG_right'].append(PoG_right_data)
                    data_dict['PoG_average'].append(PoG_average_data)
                    data_dict['gaze_angles_left'].append(gaze_angles_left_data)
                    data_dict['gaze_angles_right'].append(gaze_angles_right_data)
                    data_dict['head_angles'].append(theta_data)
                    if calibration:
                        data_dict['PoG_calib'].append(PoG_average_calib)
                        data_dict['PoG_calib2'].append(PoG_average_calib2)

                    # create a DataFrame object
                    df = pd.DataFrame.from_dict(data_dict)
                    # all stored points of gaze
                    PoG_history = gazeEstimator._all_pog_history
                    PoG_image_history = gazeEstimator._all_pog_history_i
                    # use the last n for output smoothing
                    PoG_history_max_len = 3
                    if len(gazeEstimator._all_pog_history) > PoG_history_max_len:
                        PoG_history = gazeEstimator._all_pog_history[-PoG_history_max_len:]
                        PoG_image_history = gazeEstimator._all_pog_history_i[-PoG_history_max_len:]

                    # estimate the average pog over the last maxlen samples
                    PoG_mean_over_time = np.mean(PoG_history, axis=0)
                    PoG_mean_over_time = np.mean(PoG_image_history, axis=0)

                    # visualize pog
                    cv2.arrowedLine(context['frame_bgr'], tuple(np.round(eyeball_centre).astype(np.int32)),
                                    tuple(np.round(PoG_mean_over_time).astype(int32)), (0, 255, 255),
                                    1, cv.Line_AA, tipLength = 0.01)

                    # clear the estimations of the two eyes
                    gazeEstimator.pog[:, :] = 0
                    gazeEstimator.pog_i[:, :] = 0

                if can_use_eyelid:
                    cv2.polylines(context['frame_bgr'], [np.round(eyelid_landmarks).astype(np.int32).reshape(-1,1,2)],
                                  isClosed = True, color = (255, 0, 0), thickness = 1, lineType = cv2.LINE_AA)

                if can_use_iris:
                    cv2.polylines(context['frame_bgr'], [np.round(iris_landmarks).astype(np.int32).reshape(-1,1,2)],
                                  isClosed = True, color = (0, 255, 255), thickness = 1, lineType = cv2.LINE_AA)
                    cv2.drawMarker(
                        context['frame_bgr'], tuple(np.round(iris_centre).astype(np.int32)),
                        color = (0, 255, 255), markerType = cv2.MARKER_CROSS, markerSize = 4,
                        thickness = 1, line_type = cv2.LINE_AA)

            # draw landmarks
            for (a, b) in landmarks:
                cv2.circle(frame, (a, b), 2, (0, 0, 255), -1)

            # print the new frame with cube and axis
            utils.draw_axis(context['frame_bgr'], yaw_predicted, pitch_predicted, roll_predicted, tdx = video_width*0.4,
                            tdy = video_height*0.5, size=200) # hnet
            utils.draw_axis(context['frame_bgr'], euler_angle[1], - euler_angle[0], euler_angle[2], tdx=video_width * 0.4,
                            tdy=video_height * 0.5, size=200)  # use when hnet

            cv2.putText(context['frame_bgr'], "Hopenet Model", (150,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2,
                        cv2.LINE_AA)
            cv2.putText(context['frame_bgr'], "solvePnP", (150,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2,
                        cv2.LINE_AA)

            cv2.rectangle(context['frame_bgr'], (int(x_min), int(y_min), int(x_max), int(y_max)), (0, 255, 0), 1)
            cv2.rectangle(context['frame_bgr'], (int(x_min2), int(y_min2), int(x_max2), int(y_max2)), (0, 0, 255), 1)
            # end = time.perf_counter()
            # print(end - start)

        #debug flip frame
        frame = cv2.flip(frame,1)

        cv2.imshow("Result", frame)

        if ((cv2.waitKey(10) & 0xFF) == ord('q')):
            break

        fn = 0
        while (fn < filter):
            fn+=1
            ret, frame = video.read()

    video.release()
    cv2.destroyAllWindows()








