
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Original (gray) image

full_img = cv2.imread('gray_frame_14750.png', 0)

# Original (gray) cropped image

img = full_img[full_img.shape[0] - 185: full_img.shape[0], :]

# Blur image and filter horizontal components to make rail lines more detectable.

blur = cv2.GaussianBlur( img,(7, 7), 7 )

sobelx2 = cv2.Sobel( blur, cv2.CV_64F, 1, 0, ksize=3 )

abs_sobel64f = np.absolute( sobelx2 )
sobel_x = np.uint8( abs_sobel64f )

edges = cv2.Canny( sobel_x, 40, 200 )

# HOUGH LINES

x_start = []
x_end = []
y_start = []
y_end = []
x_0 = []
y_0 = []

# Search lines in the cropped image

lines = cv2.HoughLines( edges, 1, np.pi/180, 150 )

# Convert points into the original image coordinates

L = 500
thresh = 200

for rho,theta in lines[:, 0, :]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + L*(-b))
    y1 = int(y0 + L*(a)) + np.size(full_img, 0) - np.size(img, 0)
    x2 = int(x0 - L*(-b))
    y2 = int(y0 - L*(a)) + np.size(full_img, 0) - np.size(img, 0)

    # x threshold on edge segments to detect the ones representing rails
    if x1 > thresh:
        x_0.append(x0)
        y_0.append(y0)
        x_start.append(x1)
        y_start.append(y1)
        x_end.append(x2)
        y_end.append(y2)

m = []
c = []

# Computation of the expression of straight lines y = mx + c and their interception point (X_i, Y_i)

m.append((float(y_end[0] - y_start[0])/(x_end[0] - x_start[0])))
c.append(float(y_end[0] - m[0]*x_end[0]))

m.append((float(y_end[1] - y_start[1])/(x_end[1] - x_start[1])))
c.append(float(y_end[1] - m[1]*x_end[1]))

Y_i = int((m[0]*c[1] - m[1]*c[0])/(m[0] - m[1]))
X_i = int((Y_i - c[0])/m[0])

print 'LINE 1 m c ', m[0], c[0], '\n', 'LINE 2 m c ', m[1], c[1]

# Change edge points

for l in range(len(x_start) - 1):
    y_start[l] = np.size(full_img, 0)
    x_start[l] = int((y_start[l] - c[l]) / m[l])

    y_end[l] = 10
    x_end[l] = int((y_end[l] - c[l]) / m[l])

for i in range(len(x_start) - 1):

    cv2.line( full_img, (x_start[i], y_start[i]), (x_end[i], y_end[i]), (255, 255, 255), 2 )

    cv2.drawMarker(full_img, (X_i, Y_i), (0,0,255),
                   markerType=cv2.MARKER_CROSS,
                   markerSize=20,
                   thickness=2,
                   line_type=cv2.LINE_AA)

    cv2.drawMarker(full_img, (x_start[i], y_start[i]), (0, 0, 255),
                   markerType=cv2.MARKER_STAR,
                   markerSize=20,
                   thickness=2,
                   line_type=cv2.LINE_AA)

    cv2.drawMarker(full_img, (x_end[i], y_end[i]), (0, 0, 255),
                   markerType=cv2.MARKER_TRIANGLE_UP,
                   markerSize=20,
                   thickness=2,
                   line_type=cv2.LINE_AA)

#cv2.imwrite('lines_full.png', full_img)

print 'X0', x_0, '\nY0', y_0, '\nX1', x_start, '\nY1', y_start,'\nX2', x_end, '\nY2', y_end, '\n'
print 'Focus of expansion coordinates [x, y] =', X_i,'', Y_i

full_img_wLines = full_img
plt.imshow( full_img_wLines, cmap = 'gray' )
plt.title( 'Line extraction' ), plt.xticks([]), plt.yticks([])

plt.show()