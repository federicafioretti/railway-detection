
import cv2
import numpy as np
from matplotlib import pyplot as plt

class Rail:
    def __init__(self, xstart, ystart, w, h):
        self.xstart = xstart
        self.ystart = ystart
        self.w = w
        self.w0 = w
        self.h = h
        self.scale = 0.95
        self.railX = [xstart]
        self.railY = [ystart]

    # returns the lower image coordinates where the rails start
    def get_rail(self):
            return self.railX, self.railY

    # returns the template
    def get_template(self, frame, pos):
        template = frame[(self.railY[pos]-self.h):self.railY[pos], self.railX[pos]:(self.railX[pos] + self.w)]
        return template

    # returns upper stripe of image to be processed with respect to the current template
    def get_nextrow(self, frame, pos):
        row = frame[(self.railY[pos]-2*self.h):(self.railY[pos]-self.h), 0:np.size(frame, 1)]
        return row

    # keep track of the lower left corner for each extracted template, in order to reconstruct the whole rail profile
    def push(self, pos):
        self.railX.append(pos[0])
        self.railY.append(pos[1])

    # draws the rectangular bound around the template
    def mark(self, img, pos):
        cv2.rectangle(img, (self.railX[pos], self.railY[pos]), (self.railX[pos]+self.w, self.railY[pos]-self.h), (255,255,255), 1)

    # match the given template with the immediately above one within the same frame, having same height and larger width
    def find_next(self, img, pos, MAX, method=0, plotres=0, weights_on=1):
        self.w = int(self.w0 * self.scale**pos)
        # methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF_NORMED]
        tmpl = self.get_template(img, pos)
        row = self.get_nextrow(img, pos)

        # definition of the upper template to be analysed
        startx = self.railX[pos]-2*self.w

        if startx <= 0:
            startx =0

        endx = startx+4*self.w

        if endx >= img.shape[1]:
            endx = img.shape[1]


        # Correlation is weighted by a Lorentzian function centred at the peak of correlation given by the lower left corner
        # of the previously extracted template
        xcorr = cv2.matchTemplate(row[0:self.h, startx:endx], tmpl, method=cv2.TM_CCOEFF_NORMED)

        a = 0.001*(pos*2+1) # set Lorentzian shape
        xcorrW = np.zeros_like(xcorr)
        L = []
        val = []
        val.extend(range(0, np.size(xcorr[0])))

        for i in range(0, np.size(xcorr,1)):
            L.append(1/(1 + a*pow(val[i] - MAX, 2)))
            xcorrW[0][i] = L[i]*xcorr[0][i]

        min_val0, max_val0, min_loc0, max_loc0 = cv2.minMaxLoc(xcorr)

        if weights_on == 1:
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(xcorrW)
        else:
            max_val = max_val0
            max_loc = max_loc0

        if pos == 0:
            max_loc = max_loc0

        test_img = cv2.cvtColor(row[0:self.h, startx:startx+np.size(xcorr[0])], cv2.COLOR_GRAY2BGR)

        # view results of correlation analysis
        if plotres == 1:
            plt.subplot(211)
            plt.imshow(test_img)
            plt.subplot(212)
            plt.plot(val, xcorr[0], 'y')
            plt.plot(val, xcorrW[0], 'b')
            plt.plot(val, L, 'g')
            plt.plot(max_loc0[0], max_val0, 'ro')
            plt.axis([0, np.size(xcorr[0]), min_val, 1.1])
            plt.title('Result, iter:' + str(pos)), plt.xticks([0, np.size(xcorr[0])]), plt.yticks([0, max_val])
            plt.show()

        return startx+max_loc[0], self.railY[pos]-self.h, max_loc

    # rail detection in the lower part of the image, using Hough Lines, necessary before starting the iterative rail
    # tracking using template matching
    def extract_rail_line(self, frame, start_rows, end_rows, start_cols, end_cols,canny_min, canny_max, hough_I, length,
                          theta_min=0.0, theta_max=1.0, expt_start=600, max_displacement=15):

        # elaborate lower part of the image containing the rails to make lines more visible
        patch = frame[start_rows:end_rows, start_cols:end_cols]
        patch_flipped = cv2.flip(patch, 0)
        gray_patch = cv2.cvtColor(patch_flipped, cv2.COLOR_BGR2GRAY)
        blur1 = cv2.GaussianBlur(gray_patch, (7, 7), 7)
        sobelx2 = cv2.Sobel(blur1, cv2.CV_64F, 1, 0, ksize=3)

        abs_sobel64f = np.absolute(sobelx2)
        sobel_x = np.uint8(abs_sobel64f)

        rails_edges = cv2.Canny(sobel_x, canny_min, canny_max)

        k = cv2.waitKey(1)
        if k == 115:
            while 1:
                k = cv2.waitKey(10)
                if k == 115:
                    break

        # list to contain edges of detected lines with Hough Transform

        xr_start = []
        xr_end = []
        yr_start = []
        yr_end = []
        x_0 = []
        y_0 = []

        lines = cv2.HoughLines(rails_edges, 1, np.pi/180, hough_I)

        # min displacement for the line edge with respect to the init position
        min_delta = 20
        min_val = expt_start

        # search for lines according to Hough Transform
        for rho, theta in lines[:, 0, :]:

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + length * (-b))
            y1 = int(y0 + length * (a))
            x2 = int(x0 - length * (-b))
            y2 = int(y0 - length * (a))

            # check whether the direction of the detected lines is coherent with the direction expected

            if theta < theta_max and theta > theta_min and abs(rho/np.cos(theta)+start_cols - expt_start) < max_displacement:

                x_0.append(x0)
                y_0.append(y0)
                xr_start.append(x1)
                yr_start.append(y1)
                xr_end.append(x2)
                yr_end.append(y2)

                # computes the closest detected line to the expected one and updates the edge position for next frame
                [min_delta, min_val] = self.nearest_line(rho, theta, start_cols, expt_start, min_delta, min_val)

        expt_start = min_val

        # trace lines on lower patch
        for l in range(len(xr_start) - 1):
            patch_wlines = cv2.line(patch_flipped, (xr_start[l], yr_start[l]), (xr_end[l], yr_end[l]), (255, 255, 255), 2)

        return patch_wlines, start_cols, end_cols, expt_start

    # determines the line detected among the set returned by Hough Lines
    def nearest_line(self, rho, theta, start_cols, expt_start, min_delta, min_val):

        if abs(rho / np.cos(theta) + start_cols - expt_start) < min_delta:
            min_delta = abs(rho / np.cos(theta) + start_cols - expt_start)
            min_val = rho / np.cos(theta) + start_cols
        return min_delta, min_val

    # show patch containing the railway on the lower part of image
    def patch_wRails(self, watch, patchL, patchR):

        patchL[0:np.size(patchR, 0), 250:np.size(patchR, 1)] = patchR[0:np.size(patchR, 0), 250:np.size(patchR, 1)]
        patch = cv2.flip(patchL, 0)

        if watch == 1:
            cv2.imshow('frame', patch)

