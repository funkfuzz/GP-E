import numpy as np


# ToDo: implement a csv or yml reading function to get new params directly from a csv/yml file
video_width = 640
video_height = 480
# webcam data from calibration with chAruco
# scaled focal length
focal_x = 430
focal_y = 620

cam_intrinsic_mat = np.array(
    [[focal_x, 0, video_width / 2],
     [0, focal_y, video_height / 2],
     [0, 0, 1]], dtype = "int"
)

# distortion coeffs Webcam
dist_coeffs = np.array([-0.3, 0.15, 0.0, 0.0, -0.2])