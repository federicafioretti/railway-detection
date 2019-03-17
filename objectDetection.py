
import cv2
import numpy as np


class RailwayObjectDetection:
    def __init__(self, ml_rail, cl_rail, mr_rail, cr_rail, focusOfExpansion):
        self.ml_rail = ml_rail
        self.cl_rail = cl_rail
        self.mr_rail = mr_rail
        self.cr_rail = cr_rail
        self.foe = focusOfExpansion

    # assess the turn markings detection
    def check_for_turn(self, bbox, bboxold, x, y, x_old, y_old, frame, mask, dynFOEx, pole_points, pole_oldpoints):

        # distance between matched points along x and y
        distx = np.abs(x - x_old)
        disty = np.abs(y - y_old)

        # dynamic FOE variable is initialised as the straight track value, for each feature point considered
        foe1 = self.foe[0]

        # check for keypoints matching goodness
        for (a, b, w, h) in bbox:
            for (aold, bold, wold, hold) in bboxold:
                cv2.rectangle(frame, (a, b), (a + w, b + h), (0, 255, 255), 2)
                if ((x > a and x < (a + w) and y > b and y < (b + h)) or
                        (x_old > aold and x_old < (aold + wold) and y_old > bold and y_old < (bold + hold))) and \
                                distx < 30 and disty < 30:

                    cv2.line(mask, (int(x), int(y)), (int(x_old), int(y_old)), (255, 255, 255), 2)

                    pole_points.append((x,y))
                    pole_oldpoints.append((x_old, y_old))

                    # significant turn condition
                    if abs(x - x_old) > 2:

                        # compute which direction the optical flow should have
                        mt = (float(y - y_old) / (x - x_old + 0.0001)) + 0.0001
                        ct = (float(y - mt * x))

                        # compute new x coord for focus of expansion FOE
                        foex = float((self.foe[1] - ct) / mt)
                        foe1 = foe1 + (foex - foe1)

                        # arrange all FOE x coord in array, making sure that foe1 result is valid
                        if not np.isnan(foe1):
                            dynFOEx.append(foe1)

        return dynFOEx, pole_points, pole_oldpoints

    # assess the kilometer signs detection
    def get_km_features(self, bbox, bboxold, x, y, x_old, y_old, points_km, oldpoints_km):
        distx = np.abs(x - x_old)
        disty = np.abs(y - y_old)

        # check for keypoints matching goodness
        for (a, b, w, h) in bbox:
            for (aold, bold, wold, hold) in bboxold:
                if ((x > a and x < (a + w) and y > b and y < (b + h)) or
                        (x_old > aold and x_old < (aold + wold) and y_old > bold and y_old < (bold + hold))) and \
                                distx < 30 and distx > 12 and disty < 30 and disty > 8:
                    points_km.append((float(x),float(y)))
                    oldpoints_km.append((float(x_old), float(y_old)))
        return points_km, oldpoints_km
