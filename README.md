Railway detection
=============

The algorithm described here performs the task of discriminating the conditions of straight and turn motion using the only visual input. 


The extraction of the portion of image occupied by the rails is handled by *Template Matching* algorithm.\
The *Railway Extraction* algorithm process each frame of the video, starting with the extraction of the rail lines. \
In order to make the structure of the environment sufficiently visible, a proper elaboration is needed to filter unwanted details out of the image. Precisely each capture undergoes the following steps:

* Gray scale conversion
* Gaussian blur through a square kernel of size 7, with standard deviation of 7 pixels in x direction.
* Sobel on the x-axis (y-axis) of the image to make vertical lines (horizontal lines) more detectable. Then the absolute value of the resulting image is taken, in order to equally take into account positive and negative gradients
* Canny application to obtain the binary image only containing the pixels recognized as edges
* Hough transform to gather the lines of interest.

This process operates on the lower part of the image, where straight lines are a good approximation of the imaged rails. 
After selecting the image patches containing left and right rails, the line extraction has been done using the OpenCV procedures for the Hough Lines transform and the image gradient computation Canny. 

In order to extract only the lines of interest, several conditions have been defined on the ρ and θ parameters of the Hough Transform, given as result of the OpenCV HoughLines step. 

The guessed position for the lower edges of these segments gets updated frame after frame, and a minimum displacement along x-axis has been defined to exclude too far lines.\
The identification of the lower part of the imaged rails allows for a good initialization of the template matching phase: for each rail, the first template is extracted according to the position guessed by the *Railway lines extraction*. A correlation analysis is done between the template and the upper stripe of image.

Naming:
* w_F, h_F the frame width and height
* w_T, h_T the template width and height
* w_I = 2w_T the test image width
* h_I = h_T the test image height
* ulC_T the upper left corner coordinates of the template
* ulC_I the upper left corner coordinates of the test image
* maxcorrX the x-coordinate of the point within the test image having the highest value of correlation with the template.

The current frame of the video is analyzed to look for the rails profile, by updating the upper left corner of both the template and test image for a specific number of iterations (27):

* ulC_T = (maxcorrX, ulC_{T_y} - h_T)
* ulC_I = (maxcorrX - w_T, ulC_{T_y} - 2h_T)

The search for the sequence of templates fitting one of the rails also considers the alignment of the few previous ones, in order to approximate at best the curvature of the rails.

The Focus of Expansion position within the image is ideally individuated by the optical flow. In a railway scenario the environment may not be sufficiently structured and textured. 

The infrastructure has few geometric elements and sometimes high presence of vegetation. In these cases wrong feature matching may occur, with the risk of having a set of unreliable tracks of points across subsequent frames. Relying on the detection of the rails and signs extracted within an image, two separate estimates of the FoE coordinates can be obtained as:

* the intersection of the lines approximating the lower part of the rails,
with a constant horizon at a y coordinate in the image, depending on the
principal point
* the mean point of intersection among all matched points in the
bounding box of the turn markings, considering the same fixed y coordinate

When both the estimates computed in this fashion are available, the FoE position is the mean value of the coordinates.
The displacement of the FoE along the horizon line can be registered and exploited to filter wrongly matched feature as well as to gather important angular motion cues. \
