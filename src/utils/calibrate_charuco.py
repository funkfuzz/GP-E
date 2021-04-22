''' Script taken from the following tutorial:
    https://medium.com/vacatronics/3-ways-to-calibrate-your-camera-using-opencv-and-python-395528a51615

    To perform an automated calibration you can use this tutorial:
    https://docs.opencv.org/master/d7/d21/tutorial_interactive_calibration.html

'''
import os
from charuco import calibrate_charuco
from utils import load_coefficients, save_coefficients
import cv2

# Parameters
IMAGES_DIR = 'path_to_images'
IMAGES_FORMAT = 'jpg'
# Dimensions in cm
MARKER_LENGTH = 2.7
SQUARE_LENGTH = 3.2

# Calibrate
ret, mtx, dist, rvecs, tvecs = calibrate_charuco(
    IMAGES_DIR,
    IMAGES_FORMAT,
    MARKER_LENGTH,
    SQUARE_LENGTH
)
# Save coefficients into a file
save_coefficients(mtx, dist, os.path.dirname(os.path.realpath(__file__)) +'/../calibration/calibration_charuco.yml')

# Load coefficients
mtx, dist = load_coefficients(os.path.dirname(os.path.realpath(__file__)) +'/../calibration/calibration_charuco.yml')
original = cv2.imread('path_to_image')
dst = cv2.undistort(original, mtx, dist, None, mtx)
cv2.imwrite('undist_charuco.jpg', dst)