import cv2
import numpy as np


# FILTER MATCHED POINT ACCORDING TO THE OPTICAL FLOW DIRECTION
# optical flow direction computation, to check the goodness of a matched pair of points
def filter_features_matching(foenew, x, x_old, y, y_old, mask):

    # direction connecting the old feature coordinates and the current focus of expansion foenew
    m = (float(foenew[1] - y_old) / (foenew[0] - x_old))
    c = (float(foenew[1] - m * foenew[0]))

    # displacement between matched points along x and y
    distx = np.abs(x - x_old)
    disty = np.abs(y - y_old)

    # check whether the feature displacement is parallel to the direction connecting its actual coordinates
    # to the current focus of expansion position
    check = abs(y - m * x - c)

    if check < 2 and distx < 30 and disty < 30:
        cv2.line(mask, (int(x), int(y)), (int(x_old), int(y_old)), (255, 255, 255), 1)
