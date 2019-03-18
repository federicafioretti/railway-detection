Railway detection
=============
The railway analysis performed by this program (in Python 2.7, using OpenCV 3.3) addresses the following tasks:
* Detection of the turn markings and kilometer signs (Italian Railway) thanks to the **HAAR
cascade classifiers** previously trained using OpenCV.
* Rail lines detection and tracking using **HoughLines** together with an iterative 
**Template matching** process.
* *Focus Of Expansion* coordinates determined by computing the intersection between
 the straight line approximation of the rails. Moreover the optical flow within the 
 bounding box enclosing the detected turn markings is assessed to the same aim.
* Wrongly matched keypoints filtering, thanks to the *Focus Of Expansion* tracking, evaluating their optical
 flow direction, displayed in white for the complete scenario.

![Alt Text](https://raw.githubusercontent.com/federicafioretti/railway-detection/master/image-readme/execution.png)

The extraction of the portion of image occupied by the rails is handled by the Template Matching algorithm.\
The *Railway Extraction* algorithm process each frame of the video, starting with the extraction of the rail lines. \
In order to make the structure of the environment sufficiently visible, a proper elaboration is needed to filter unwanted
 details out of the image. Precisely each capture undergoes the following steps:

* Gray scale conversion
* Gaussian blur through a square kernel of size 7, with standard deviation of 7 pixels in x direction.
* Sobel on the x-axis (y-axis) of the image to make vertical lines (horizontal lines) more detectable. Then the absolute
 value of the resulting image is taken, in order to equally take into account positive and negative gradients
* Canny application to obtain the binary image only containing the pixels recognized as edges
* Hough transform to gather the lines of interest.

This process operates on the lower part of the image, where straight lines are a good approximation of the imaged rails. 
After selecting the image patches containing left and right rails, the line extraction has been done using the OpenCV 
procedures for the Hough Lines transform and the image gradient computation Canny. 
```
def extract_rail_line
    patch = frame[start_rows:end_rows, start_cols:end_cols]
    patch_flipped = cv2.flip(patch, 0)
    gray_patch = cv2.cvtColor(patch_flipped, cv2.COLOR_BGR2GRAY)
    blur1 = cv2.GaussianBlur(gray_patch, (7, 7), 7)
    sobelx2 = cv2.Sobel(blur1, cv2.CV_64F, 1, 0, ksize=3)

    abs_sobel64f = np.absolute(sobelx2)
    sobel_x = np.uint8(abs_sobel64f)

    rails_edges = cv2.Canny(sobel_x, canny_min, canny_max)
```

In order to extract only the lines of interest, several conditions have been defined on the ρ and θ parameters of the Hough 
Transform, given as result of the OpenCV HoughLines step. 

The patch containing the rails in the lower image has been **flipped** to compute the
expected the guess for x coordinate `min_val` to extract the first template containing a rail 
element within the next frame to be processed as:

min_val = ρ * cos (θ) + full_img_scale

where ρ encodes the distance from the origin of the image axis to the normal straight line 
to the implied line and θ the angle defining the slope of the normal.

The guessed position for the lower edges of these segments gets updated frame after frame, and a minimum displacement 
along x-axis has been defined to exclude too far lines, in this fashion:
```
def nearest_line

    if abs(rho / np.cos(theta) + start_cols - expt_start) < min_delta:
        min_delta = abs(rho / np.cos(theta) + start_cols - expt_start)
        min_val = rho / np.cos(theta) + start_cols
    return min_delta, min_val
```

The identification of the lower part of the imaged rails allows for a good initialization of the template matching phase: 
for each rail, the first template is extracted according to the position guessed by the *Railway lines extraction*. 
A weighted correlation analysis is done between the template and the upper stripe of image.

    xcorr = cv2.matchTemplate(row[0:self.h, startx:endx], 
                              tmpl, 
                              method=cv2.TM_CCOEFF_NORMED)

    a = 0.001*(pos*2+1) # set Lorentzian shape
    xcorrW = np.zeros_like(xcorr)
    L = []
    val = []
    val.extend(range(0, np.size(xcorr[0])))

    for i in range(0, np.size(xcorr,1)):
        L.append(1/(1 + a*pow(val[i] - MAX, 2)))
        xcorrW[0][i] = L[i]*xcorr[0][i]


Naming:
* w_F, h_F the frame width and height
* w_T, h_T the template width and height
* w_I = 2w_T the test image width
* h_I = h_T the test image height
* ulC_T the upper left corner coordinates of the template
* ulC_I the upper left corner coordinates of the test image
* maxcorrX the x-coordinate of the point within the test image having the highest value of correlation with the template.

ulC_T = (maxcorrX, ulC_T_y - h_T) \
ulC_I = (maxcorrX - w_T, ulC_T_y - 2h_T)

The current frame of the video is analyzed to look for the rails profile, by updating the upper left corner of both the template and test image for a specific number of iterations (27):

```
loop = 1
while loop < 27:
    xl, yl, Ml = railL.find_next(matchImage, 
                                loop - 1, 
                                Ml[0], 
                                plotres=watch_xcorr,
                                weights_on=weighted_version)
    xr, yr, Mr = railR.find_next(matchImage, 
                                loop - 1, 
                                Mr[0], 
                                plotres=watch_xcorr,
                                weights_on=weighted_version)
    railL.push((xl, yl))
    railR.push((xr, yr))
    railL.mark(frame, loop)
    railR.mark(frame, loop)
```
The infrastructure has few geometric elements and sometimes high presence of vegetation. In these cases wrong feature matching may occur, with the risk of having a set of unreliable tracks of points across subsequent frames. Relying on the detection of the rails and signs extracted within an image, two separate estimates of the FoE coordinates can be obtained as:

* the intersection of the lines approximating the lower part of the rails,
with a constant horizon at a y coordinate in the image, depending on the
principal point
* the mean point of intersection among all matched points in the
bounding box of the turn markings, considering the fixed horizon y coordinate (given by 
camera calibration, i.e. the principal point coordinates)

When both the estimates computed in this fashion are available, the FoE position is the mean value of the coordinates.
The displacement of the FoE along the horizon line can be registered and exploited to filter wrongly matched feature and may provide important motion cues.
