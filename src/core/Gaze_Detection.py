import os
import cv2 as cv
import tensorflow as tf
import glob
import numpy as np
import scipy
from scipy import optimize


class Gaze_Detection:
    """ The class that performs the gaze estimation.

    This class uses a pre-trained neural network model called GazeML to
    estimate 18 eye landmarks. These are needed for the estimation of the
    gaze direction. The class has two methods implemented for the compu-
    tation of the gaze direction. The first one is a simple method, yet fast.
    The second is more computationally intensive, yet more precise. The gaze-
    direction information can then be used in combination with head-pose information
    to estimate the person's point of gaze by the get_pog() method of this class.

    """
    def __init__(self, tensorflow_Session = tf.Session):
        self.sess = tensorflow_Session # ToDo: update private and non-private members
        self._data_format = 'NCHW'
        self._dir = os.path.dirname(os.path.realpath(__file__))+'/../models'
        self._pb = glob.glob('%s/gaze.pb'% (self._dir))
        self.eye, self.heatmaps, self.landmarks, self.radius = self._get_tensor_names()
        self.batch_size = 2
        self.all_gaze_histories = []
        self.pog = np.zeros((3, 2), dtype='float32')
        self.pog_i = np.zeros((2, 2), dtype='float32')
        self._all_pog_history = []
        self._all_pog_history_i = []
        self._pose_mat = []
        self._rot_matrix = []
        self._translation_vec = []
        self._intr_mat= [] # camera intrinsic matrix
        self._3d_head = []
        self._frame_shape= []
        self.eye_landmarks = []


    def _load_frozen_tf_model(self):
        """ Loads the frozen TF model (gazeML.pb)"""
        with tf.gfile.FastGFile(self._pb[0], 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

            # Fix nodes of freezed model
            for node in graph_def.node:
                if node.op == 'RefSwitch':
                    node.op = 'Switch'
                    for index in range(len(node.input)):
                        if 'moving_' in node.input[index]:
                            node.input[index] = node.input[index] + '/read'
                elif node.op == 'AssignSub':
                    node.op = 'Sub'
                    if 'use_locking' in node.attr: del node.attr['use_locking']

            # Export corrected fixed freezed model pb file.
            with tf.gfile.FastGFile('./gaze_better.pb', mode='wb') as model_fixed:
                model_fixed.write(graph_def.SerializeToString())

            tf.import_graph_def(graph_def, name='')

    def _get_tensor_names(self):
        #self._load_frozen_tf_model()

        """ Loads the frozen TF model (gazeML.pb)"""
        with tf.gfile.FastGFile(self._pb[0], 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
        # assign names for input and output tensors of the TF model
        # input tensor
        eye = self.sess.graph.get_tensor_by_name('Webcam/fifo_queue_DequeueMany:1')
        # eye_index = sess.graph.get_tensor_by_name('Webcam/fifo_queue_DequeueMany:2')
        # output tensors
        heatmaps = self.sess.graph.get_tensor_by_name('hourglass/hg_2/after/hmap/conv/BiasAdd:0') # for GazeML with 3 hg modules 'hourglass/hg_3/after/hmap/conv/BiasAdd:0'
        landmarks = self.sess.graph.get_tensor_by_name('upscale/mul:0')
        radius = self.sess.graph.get_tensor_by_name('radius/out/fc/BiasAdd:0')
        self.sess.run(tf.global_variables_initializer())

        return eye, heatmaps, landmarks, radius

    # ToDo implement the SVR method used by GazeML

    def get_gaze_direction_simple(self, iris_c, eyeball_c, eyeball_r):
        """ A simple yet fast approach to estimate the gaze direction.

        The method builds a ray from the eyeball center through the iris center to estimate
        the gaze direction.

        :param iris_c: iris centre
        :param eyeball_c: eyeball centre
        :param eyeball_r: eyeball radius
        :return: gaze direction vector as pitch and yaw angles
        """

        i_x0, i_y0 = iris_c
        e_x0, e_y0 = eyeball_c
        theta = -np.arcsin(np.clip((i_y0 - e_y0) / eyeball_r, -1.0, 1.0))
        phi = np.arcsin(np.clip((i_x0 - e_x0) / (eyeball_r * -np.cos(theta)),
                                -1.0, 1.0))

        return np.array([theta, phi])

    def get_gaze_direction_complex(self, iris, iris_c, eyeball_c, eyeball_r, initial_gaze = None):
        """ A more precise, yet more complex and computationally heavy approach for the estimation of the gaze direction.


        The function fits a 3D model of the human eye to the input eye landmarks to estimate the gaze direction.

        :param iris: iris edge landmarks
        :param iris_c: iris centre
        :param eyeball_c: eyeball centre
        :param eyeball_r: eyeball radius
        :param initial_gaze: initial gaze direction if provided
        :return: gaze direction vector as pitch and yaw angles
        """

        i_x0, i_y0 = iris_c
        e_x0, e_y0 = eyeball_c

        if initial_gaze is not None:
            theta, phi = initial_gaze
            # theta = -theta
        else:
            theta = np.arcsin(np.clip((i_y0 - e_y0) / eyeball_r, -1.0, 1.0))
            phi = np.arcsin(np.clip((i_x0 - e_x0) / (eyeball_r * -np.cos(theta)), -1.0, 1.0))

        delta = 0.1 * np.pi
        if iris[0, 0] < iris[4, 0]:  # flipped
            alphas = np.flip(np.arange(0.0, 2.0 * np.pi, step=np.pi / 4.0), axis=0)
        else:
            alphas = np.arange(-np.pi, np.pi, step=np.pi / 4.0) + np.pi / 4.0
        sin_alphas = np.sin(alphas)
        cos_alphas = np.cos(alphas)

        def gaze_fit_loss_func(inputs):
            theta, phi, delta, phase = inputs
            sin_phase = np.sin(phase)
            cos_phase = np.cos(phase)
            # sin_alphas_shifted = np.sin(alphas + phase)
            sin_alphas_shifted = sin_alphas * cos_phase + cos_alphas * sin_phase
            # cos_alphas_shifted = np.cos(alphas + phase)
            cos_alphas_shifted = cos_alphas * cos_phase - sin_alphas * sin_phase

            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)
            sin_delta_sin = np.sin(delta * sin_alphas_shifted)
            sin_delta_cos = np.sin(delta * cos_alphas_shifted)
            cos_delta_sin = np.cos(delta * sin_alphas_shifted)
            cos_delta_cos = np.cos(delta * cos_alphas_shifted)
            # x = -np.cos(theta + delta * sin_alphas_shifted)
            x1 = -cos_theta * cos_delta_sin + sin_theta * sin_delta_sin
            # x *= np.sin(phi + delta * cos_alphas_shifted)
            x2 = sin_phi * cos_delta_cos + cos_phi * sin_delta_cos
            x = x1 * x2
            # y = np.sin(theta + delta * sin_alphas_shifted)
            y1 = sin_theta * cos_delta_sin
            y2 = cos_theta * sin_delta_sin
            y = y1 + y2

            ix = e_x0 + eyeball_r * x
            iy = e_y0 + eyeball_r * y
            dx = ix - iris[:, 0]
            dy = iy - iris[:, 1]
            out = np.mean(dx ** 2 + dy ** 2)

            # In addition, match estimated and actual iris centre
            iris_dx = e_x0 + eyeball_r * -cos_theta * sin_phi - i_x0
            iris_dy = e_y0 + eyeball_r * sin_theta - i_y0
            out += iris_dx ** 2 + iris_dy ** 2

            # sin_alphas_shifted = sin_alphas * cos_phase + cos_alphas * sin_phase
            # cos_alphas_shifted = cos_alphas * cos_phase - sin_alphas * sin_phase
            dsin_alphas_shifted_dphase = -sin_alphas * sin_phase + cos_alphas * cos_phase
            dcos_alphas_shifted_dphase = -cos_alphas * sin_phase - sin_alphas * cos_phase

            # sin_delta_sin = np.sin(delta * sin_alphas_shifted)
            # sin_delta_cos = np.sin(delta * cos_alphas_shifted)
            # cos_delta_sin = np.cos(delta * sin_alphas_shifted)
            # cos_delta_cos = np.cos(delta * cos_alphas_shifted)
            dsin_delta_sin_ddelta = cos_delta_sin * sin_alphas_shifted
            dsin_delta_cos_ddelta = cos_delta_cos * cos_alphas_shifted
            dcos_delta_sin_ddelta = -sin_delta_sin * sin_alphas_shifted
            dcos_delta_cos_ddelta = -sin_delta_cos * cos_alphas_shifted
            dsin_delta_sin_dphase = cos_delta_sin * delta * dsin_alphas_shifted_dphase
            dsin_delta_cos_dphase = cos_delta_cos * delta * dcos_alphas_shifted_dphase
            dcos_delta_sin_dphase = -sin_delta_sin * delta * dsin_alphas_shifted_dphase
            dcos_delta_cos_dphase = -sin_delta_cos * delta * dcos_alphas_shifted_dphase

            # x1 = -cos_theta * cos_delta_sin + sin_theta * sin_delta_sin
            # x2 = sin_phi * cos_delta_cos + cos_phi * sin_delta_cos
            dx1_dtheta = sin_theta * cos_delta_sin + cos_theta * sin_delta_sin
            dx2_dtheta = 0.0
            dx1_dphi = 0.0
            dx2_dphi = cos_phi * cos_delta_cos - sin_phi * sin_delta_cos
            dx1_ddelta = -cos_theta * dcos_delta_sin_ddelta + sin_theta * dsin_delta_sin_ddelta
            dx2_ddelta = sin_phi * dcos_delta_cos_ddelta + cos_phi * dsin_delta_cos_ddelta
            dx1_dphase = -cos_theta * dcos_delta_sin_dphase + sin_theta * dsin_delta_sin_dphase
            dx2_dphase = sin_phi * dcos_delta_cos_dphase + cos_phi * dsin_delta_cos_dphase

            # y1 = sin_theta * cos_delta_sin
            # y2 = cos_theta * sin_delta_sin
            dy1_dtheta = cos_theta * cos_delta_sin
            dy2_dtheta = -sin_theta * sin_delta_sin
            dy1_dphi = 0.0
            dy2_dphi = 0.0
            dy1_ddelta = sin_theta * dcos_delta_sin_ddelta
            dy2_ddelta = cos_theta * dsin_delta_sin_ddelta
            dy1_dphase = sin_theta * dcos_delta_sin_dphase
            dy2_dphase = cos_theta * dsin_delta_sin_dphase

            # x = x1 * x2
            # y = y1 + y2
            dx_dtheta = dx1_dtheta * x2 + x1 * dx2_dtheta
            dx_dphi = dx1_dphi * x2 + x1 * dx2_dphi
            dx_ddelta = dx1_ddelta * x2 + x1 * dx2_ddelta
            dx_dphase = dx1_dphase * x2 + x1 * dx2_dphase
            dy_dtheta = dy1_dtheta + dy2_dtheta
            dy_dphi = dy1_dphi + dy2_dphi
            dy_ddelta = dy1_ddelta + dy2_ddelta
            dy_dphase = dy1_dphase + dy2_dphase

            # ix = w_2 + eyeball_radius * x
            # iy = h_2 + eyeball_radius * y
            dix_dtheta = eyeball_r * dx_dtheta
            dix_dphi = eyeball_r * dx_dphi
            dix_ddelta = eyeball_r * dx_ddelta
            dix_dphase = eyeball_r * dx_dphase
            diy_dtheta = eyeball_r * dy_dtheta
            diy_dphi = eyeball_r * dy_dphi
            diy_ddelta = eyeball_r * dy_ddelta
            diy_dphase = eyeball_r * dy_dphase

            # dx = ix - iris_landmarks[:, 0]
            # dy = iy - iris_landmarks[:, 1]
            ddx_dtheta = dix_dtheta
            ddx_dphi = dix_dphi
            ddx_ddelta = dix_ddelta
            ddx_dphase = dix_dphase
            ddy_dtheta = diy_dtheta
            ddy_dphi = diy_dphi
            ddy_ddelta = diy_ddelta
            ddy_dphase = diy_dphase

            # out = dx ** 2 + dy ** 2
            dout_dtheta = np.mean(2 * (dx * ddx_dtheta + dy * ddy_dtheta))
            dout_dphi = np.mean(2 * (dx * ddx_dphi + dy * ddy_dphi))
            dout_ddelta = np.mean(2 * (dx * ddx_ddelta + dy * ddy_ddelta))
            dout_dphase = np.mean(2 * (dx * ddx_dphase + dy * ddy_dphase))

            # iris_dx = e_x0 + eyeball_radius * -cos_theta * sin_phi - i_x0
            # iris_dy = e_y0 + eyeball_radius * sin_theta - i_y0
            # out += iris_dx ** 2 + iris_dy ** 2
            dout_dtheta += 2 * eyeball_r * (sin_theta * sin_phi * iris_dx + cos_theta * iris_dy)
            dout_dphi += 2 * eyeball_r * (-cos_theta * cos_phi * iris_dx)

            return out, np.array([dout_dtheta, dout_dphi, dout_ddelta, dout_dphase])

        phase = 0.02
        result = scipy.optimize.minimize(gaze_fit_loss_func, x0=[theta, phi, delta, phase],
                                         bounds=(
                                             (-0.4 * np.pi, 0.4 * np.pi),
                                             (-0.4 * np.pi, 0.4 * np.pi),
                                             (0.01 * np.pi, 0.5 * np.pi),
                                             (-np.pi, np.pi),
                                         ),
                                         jac=True,
                                         tol=1e-6,
                                         method='TNC',
                                         options={
                                             # 'disp': True,
                                             'gtol': 1e-6,
                                             'maxiter': 100,
                                         })
        if result.success:
            theta, phi, delta, phase = result.x
            theta_d = (theta*180)/np.pi
            phi_d = (phi * 180) / np.pi
            print('Theta is: %s , Phi is: %s'%(theta_d,phi_d))

        return np.array([-theta, phi])

    def _landmarks_from_heatmaps(self, face, eye_index):
        """ Picks the eye landmark pixel coordinates from the heatmaps output by the GazeML model.

        :param face: the face roi
        :param eye_index: 0 for left, 1 for right eye
        :return: eye labdmarks, eye side, bool variables regarding the succesful landmarks estimation
        """

        oheatmaps, olandmarks, oradius = face['gaze']
        eyes = face['eyes']
        # Decide which landmarks are usable
        heatmaps_amax = np.amax(oheatmaps[eye_index, :].reshape(-1, 18), axis=0)
        # original GazeML demo use 0.7 instead of 0.5
        can_use_eye = np.all(heatmaps_amax > 0.5)
        can_use_eyelid = np.all(heatmaps_amax[0:8] > 0.75)
        can_use_iris = np.all(heatmaps_amax[8:16] > 0.8)
        # ToDo: fix bug when commenting out the line above
        can_use_eye, can_use_eyelid, can_use_iris = True, False, False
        # Embed eye image and annotate for picture-in-picture
        eye = eyes[eye_index]
        eye_image = eye['image']
        eye_side = eye['side']
        eye_landmarks = olandmarks[eye_index, :]
        eye_radius = oradius[eye_index][0]
        if (eye_side == 'left'):
            eye_landmarks[:, 0] = eye_image.shape[1] - eye_landmarks[:, 0]
        # Transform predictions
        eye_landmarks = np.concatenate([eye_landmarks,
                                        [[eye_landmarks[-1, 0] + eye_radius,
                                          eye_landmarks[-1, 1]]]])
        eye_landmarks = np.asmatrix(np.pad(eye_landmarks, ((0, 0), (0, 1)), 'constant', constant_values=1.0))
        eye_landmarks = (eye_landmarks * eye['inv_landmarks_transform_mat'].T)[:, :2]
        eye_landmarks = np.asarray(eye_landmarks)

        return eye_landmarks, eye_side, can_use_eye, can_use_eyelid, can_use_iris


    def draw_gaze_direction(self, input_image, origin, pitchyaw, length=40.0):
        ''' Visualizes the estimated gaze direction vectors.

        The gaze direction ray is estimated by using the eyeball centre as an origin
        and the estimated gaze direction.

        :param input_image: input image
        :param origin: the eyeball centre
        :param pitchyaw: the estimated gaze direction vector
        :param length: length of the arrow
        :return: visualization of the gaze direction vector
        '''


        image_out = input_image   #ToDo: is this really needed?
        if len(image_out.shape) == 2 or image_out.shape[2] == 1:
            image_out = cv.cvtColor(image_out, cv.COLOR_GRAY2BGR)
        dx = -length * np.sin(pitchyaw[1])
        dy = -length * np.sin(pitchyaw[0])
        cv.arrowedLine(image_out, tuple(np.round(origin).astype(np.int32)),
                       tuple(np.round([origin[0] + dx, origin[1] + dy]).astype(int)), (255, 255, 128),
                       2, cv.LINE_AA, tipLength=0.2)

        return image_out


    def _eye_detection(self, face, context):
        ''' Crops the eye RoIs.

        Crops the eyes by using the estimated facial landmarks.

        :param face: face RoI
        :param context: dict containing different image streams
        :return : eye roi

        '''

        eyes = []

        # Final output dimensions
        oh, ow = (36, 60)

        landmarks = face['landmarks']

        # Segment eyes
        for corner1, corner2, is_left in [(36, 39, True), (42, 45, False)]:
        #for corner1, corner2, is_left in [(2, 3, True), (0, 1, False)]:
            x1, y1 = landmarks[corner1, :]
            x2, y2 = landmarks[corner2, :]
            eye_width = 1.5 * np.linalg.norm(landmarks[corner1, :] - landmarks[corner2, :])
            if eye_width == 0.0:
                continue
            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

            # Centre image on middle of eye
            translate_mat = np.asmatrix(np.eye(3))
            translate_mat[:2, 2] = [[-cx], [-cy]]
            inv_translate_mat = np.asmatrix(np.eye(3))
            inv_translate_mat[:2, 2] = -translate_mat[:2, 2]

            # Rotate to be upright
            roll = 0.0 if x1 == x2 else np.arctan((y2 - y1) / (x2 - x1))
            rotate_mat = np.asmatrix(np.eye(3))
            cos = np.cos(-roll)
            sin = np.sin(-roll)
            rotate_mat[0, 0] = cos
            rotate_mat[0, 1] = -sin
            rotate_mat[1, 0] = sin
            rotate_mat[1, 1] = cos
            inv_rotate_mat = rotate_mat.T

            # Scale
            scale = ow / eye_width
            scale_mat = np.asmatrix(np.eye(3))
            scale_mat[0, 0] = scale_mat[1, 1] = scale
            inv_scale = 1.0 / scale
            inv_scale_mat = np.asmatrix(np.eye(3))
            inv_scale_mat[0, 0] = inv_scale_mat[1, 1] = inv_scale

            # Centre image
            centre_mat = np.asmatrix(np.eye(3))
            centre_mat[:2, 2] = [[0.5 * ow], [0.5 * oh]]
            inv_centre_mat = np.asmatrix(np.eye(3))
            inv_centre_mat[:2, 2] = -centre_mat[:2, 2]

            # Get rotated and scaled, and segmented image
            transform_mat = centre_mat * scale_mat * rotate_mat * translate_mat
            inv_transform_mat = (inv_translate_mat * inv_rotate_mat * inv_scale_mat *
                                 inv_centre_mat)
            eye_image = cv.warpAffine(context['image_gray'], transform_mat[:2, :], (ow, oh))
            if is_left:
                eye_image = np.fliplr(eye_image)
            eyes.append({
                'image': eye_image,
                'inv_landmarks_transform_mat': inv_transform_mat,
                'side': 'left' if is_left else 'right',
            })
        face['eyes'] = eyes


    def _eye_preprocess(self, eye):
        ''' Image preprocessing for neural network input.

        :param eye: eye roi
        :return: preprocessed eye rois
        '''

        eye = cv.equalizeHist(eye)
        eye = eye.astype(np.float32)
        eye *= 2.0 / 255.0
        eye -= 1.0
        eye = np.expand_dims(eye, -1 if self._data_format == 'NHWC' else 0)

        return eye

    def _predict(self, face, context): #ToDo passing context might be unncecessary ,revise.
        """ Predicts 18 eye landmarks.

        The GazeML prediction pipeline. The function first prepares the input image for the GazeML model
        by finding the eye regions, then performing some preprocessing adjustments. Afterwards the image
        is fed forward to the GazeML model and the heatmaps, the landmarks and the radius of
        the person's eye (in pixels) are obtained.

        :param face: face roi
        :param context: dict containing different image streams
        :return: eye heatmaps, eye landmarks, eyeball radius
        """

        self._eye_detection(face, context)
        if (len(face['eyes']) != 2):
            return
        eye1 = self._eye_preprocess(face['eyes'][0]['image'])
        eye2 = self._eye_preprocess(face['eyes'][1]['image'])
        eyeI = np.concatenate((eye1, eye2), axis=0)
        eyeI = eyeI.reshape(2, 36, 60, 1)
        Placeholder_1 = self.sess.graph.get_tensor_by_name('learning_params/Placeholder_1:0')
        feed_dict = {self.eye: eyeI, Placeholder_1: False}
        oheatmaps, olandmarks, oradius = self.sess.run((self.heatmaps, self.landmarks, self.radius), feed_dict=feed_dict)

        face['gaze'] = (oheatmaps, olandmarks, oradius)

    def predict(self, context):
        for face in context['faces']:
            x, y, w, h = face['box']
            if ((w < 160) or (h < 160)):
                continue
            self._predict(face, context)


    def get_pog(self, gaze_direction, eye_side):
        """ The function is used to estimate the point of gaze.

        The function estimates the point of gaze in 3D space,
        by building a gaze direction ray and intersecting it with a known plane
        in 3D space.

        :param gaze_direction: the gaze direction in angles
        :param eye_side: left or right eye
        :return:
        """

        # convert the coordinates
        head_model_c = self._world_to_camera_coords(self._3d_head)
        gaze_direction = self._angles_to_vector(gaze_direction)
        # select the correct origin for the gaze direction ray
        if eye_side == 'left':
            eyeball_centre = head_model_c[:, 16]
        else:
            eyeball_centre = head_model_c[:, 15]
        # estimate scale factor for z = 0 (intersection with camera xy plane)
        scale = -(eyeball_centre[2] / gaze_direction.T[2])
        point_of_gaze = eyeball_centre + scale*gaze_direction.T[:]
        point_of_gaze_image = self._camera_to_image_coords(point_of_gaze, eyeball_centre)

        return point_of_gaze, point_of_gaze_image

    def get_average_pog(self, head_yaw, threshold = 20):
        ''' Estimates the point of gaze estimations of both eyes to obtain an average value.


        :param head_yaw: the estimated head yaw angle in degrees
        :param threshold: the threshold beyond only one eye should be considered as pog_average input
        :return: athe average value for the point of gaze in camera and image coords
        '''

        # input only from the left eye pog
        if head_yaw > threshold:
            pog_average = self.pog[:, 1]
            pog_average_i = self.pog_i[:, 1]
            print('yaw_h > 20°')
        # input only from the right eye pog
        elif head_yaw < -threshold:
            pog_average = self.pog[:, 0]
            pog_average_i = self.pog_i[:, 0]
            print('yaw_h < - 20°')
        # use both eyes as pog average input
        else:
            pog_average = np.mean(self.pog, axis = 1)
            pog_average_i = np.mean(self.pog_i, axis = 1)
            print('-20° < yaw_head < 20°')

        return pog_average, pog_average_i

    def _world_to_camera_coords(self, world_coords):
        """ Changes the reference frame from world to camera.

        :param world_coords: values in world coordinates
        :return: values transformed to camera coordinates
        """

        # bring to homogenous form
        world_coords = np.insert(world_coords.T, [3], 1, axis = 0)
        # transform the head model in camera coordinates
        camera_coords = self._pose_mat.dot(world_coords) #either fusion or normal

        return camera_coords

    def _angles_to_vector(self, gaze_angles):
        """ Transforms the predicted gaze angles (yaw, pitch) = (theta, phi) to a unit gaze vector.

        :param gaze_angles: gaze direction in angles
        :return: gaze direction as vector
        """

        size = gaze_angles.shape[0]
        sin = np.sin(gaze_angles)
        cos = np.cos(gaze_angles)
        out = np.empty((size, 3))
        out[:,0] = np.multiply(cos[0], sin[1])
        out[:,1] = sin[0]
        out[:,2] = np.multiply(cos[0], cos[1])

        # ToDo: eliminate the unnecessary row
        return out.T[:,0]

    def _camera_to_image_coords(self, pog_c, eyeball_centre, focal_length=3, display_width_mm = 382, display_width_pixel = 1920): # ToDo increase precision by using the correct camera parameters
        ''' Converts the point of gaze from camera (mm) to image (pixel) coordinates.

        Transforms the units from the camera reference frame (mm) to the display coordinate system (px). To perform it
        correctly, the correct focal_length, display_width_mm, and display_width_pixel are needed. These can be obtained
        either from manufacturers datasheet, or through calibration.

        :param pog_c: point of gaze in camera coordinates.
        :param eyeball_centre: eyeball centre in pixel
        :param focal_length: focal length of the camera. The default 3 mm is just an assumption for a standard laptop webcam.
        :param display_width_mm: the width of the display in mm
        :param display_width_pixel: the width of the display in pixels
        :return:
        '''

        # x,y coordinates image plane in camera coord
        x_mm = np.divide(
            np.multiply(np.subtract(pog_c[0], eyeball_centre[0]), np.subtract(eyeball_centre[2], focal_length)),eyeball_centre[2]) + eyeball_centre[0]
        y_mm = np.divide(
            np.multiply(np.subtract(pog_c[1], eyeball_centre[1]), np.subtract(eyeball_centre[2], focal_length)),eyeball_centre[2]) + eyeball_centre[1]

        # transform in pixel units
        pixels_pro_mm = display_width_pixel / display_width_mm
        u = np.multiply(x_mm, pixels_pro_mm)
        v = np.multiply(y_mm, pixels_pro_mm)

        return np.array([u,v])

