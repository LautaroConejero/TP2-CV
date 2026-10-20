import cv2
import numpy as np


def compute_disparity(left_rect, right_rect, num_disparities=192,
                      block_size=7, uniqueness_ratio=5,
                      speckle_window_size=100, speckle_range=2):
    """Calcula disparidad en pixeles a partir de dos imagenes BGR rectificadas."""
    left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * block_size ** 2,
        P2=32 * block_size ** 2,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range,
        disp12MaxDiff=1,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM,
    )

    return matcher.compute(left_gray, right_gray) / np.float32(16.0)
