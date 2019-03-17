
import cv2
import numpy as np


class RailwayObjectDetection:
    def __init__(self, ml_rail, cl_rail, mr_rail, cr_rail, focusOfExpansion):
        self.ml_rail = ml_rail
        self.cl_rail = cl_rail
        self.mr_rail = mr_rail
        self.cr_rail = cr_rail
        self.foe = focusOfExpansion

    def check_for_turn(self, bbox, bboxold, x, y, x_old, y_old, frame, mask, dynFOEx, pole_points, pole_oldpoints):

        # distance between matched points along x and y
        distx = np.abs(x - x_old)
        disty = np.abs(y - y_old)

        # dynamic FOE variable is initialised as the straight track value, for each feature point considered
        foe1 = self.foe[0]

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

    def get_km_features(self, bbox, bboxold, x, y, x_old, y_old, points_km, oldpoints_km):
        distx = np.abs(x - x_old)
        disty = np.abs(y - y_old)

        for (a, b, w, h) in bbox:
            for (aold, bold, wold, hold) in bboxold:
                if ((x > a and x < (a + w) and y > b and y < (b + h)) or
                        (x_old > aold and x_old < (aold + wold) and y_old > bold and y_old < (bold + hold))) and \
                                distx < 30 and distx > 12 and disty < 30 and disty > 8:
                    points_km.append((float(x),float(y)))
                    oldpoints_km.append((float(x_old), float(y_old)))
        return points_km, oldpoints_km

    def elaborate_matches(self, kp, kp_old, k, matches, foenew, mask, frame, poles_bbox, poles_bb_old, dynFOEx):
        q = matches[k].queryIdx  # old frame
        t = matches[k].trainIdx  # new frame

        # keypoints coordinates p1 in current frame
        x = kp[t].pt[0]
        y = kp[t].pt[1]

        p1 = np.array((x, y))

        # keypoints coordinates p2 in previous frame
        x_old = kp_old[q].pt[0]
        y_old = kp_old[q].pt[1]

        p2 = np.array((x_old, y_old))

        # distance between matched points along x and y
        distx = np.abs(x - x_old)
        disty = np.abs(y - y_old)

        dynFOEx = self.check_for_turn(poles_bbox, poles_bb_old, x, y, x_old, y_old, distx, disty, frame,
                                                 mask, dynFOEx)

        # optical flow direction computation, to check the goodness of a matched pair of points
        m = (float(foenew[1] - y_old) / (foenew[0] - x_old))
        c = (float(foenew[1] - m * foenew[0]))

        check = abs(y - m * x - c)

        if check < 2 and distx < 30 and disty < 30:
            cv2.line(mask, (int(x), int(y)), (int(x_old), int(y_old)), (255, 255, 255), 2)

        return [x, y, x_old, y_old]