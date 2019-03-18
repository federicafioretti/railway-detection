

import cv2
import numpy as np
import argparse
import imageio
from rail import Rail
from objectDetection import RailwayObjectDetection
from fcns_track_railway import tmp_match_rail_lines
from fcn_filter_matching import filter_features_matching


# read video
class ImageIOVideoCapture:
    def __init__(self,name):
        self.r = imageio.get_reader(name)
    def seek(self,frame_index):
        self.r.set_image_index(frame_index)
    # imageio is BGR
    # opencv  is RGB
    def read(self):
        r = self.r.get_next_data()
        return (r is not None,r)

# define keypoint
class KeypointsOnTrack:
    def __init__(self, kpx, kpy, col):
        self.kpx = kpx
        self.kpy = kpy
        self.col = col

# ....... prior knowledge of the scene, computed in test_straight_track_analysis.py ........

# the focus of expansion coordinates under (straight) nominal condition and corresponds to the camera principal point
# given by the calibration
foe = np.array((948, 631))

# line coefficients fitting left and right rail
ml_rail = -1.5981132075471698
mr_rail = -14.242857142857142
cl_rail = 2146.2377358490567
cr_rail = 14134.142857142857


'''------------------------------------------------MAIN-----------------------------------------------------------'''

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-I',"--input",help="input file video",default="RegistrazioneVideoTrenoVolterra.mp4")
    parser.add_argument('-O',"--output",help="input file JSON",default="output.json")
    parser.add_argument('--cascadeTurn',help="XML of cascade",default='cascadePOLES.xml')
    parser.add_argument('--cascadeKm', help="XML of cascade", default='cascadeKM.xml')
    parser.add_argument('--imgSizeX', type=int, help="image size along x", default=1920)
    parser.add_argument('--imgSizeY', type=int, help="image size along Y", default=1080)
    parser.add_argument('--tmpWidth', type=int, help="start template width", default=35)
    parser.add_argument('--tmpHeight', type=int, help="start template height", default=15)
    parser.add_argument('--startLeftRailGuess', type=int, help="Guess for determining the position within the lower part"
                                                               " of the image of the left rail", default=630)
    parser.add_argument('--startRightRailGuess', type=int, help="Guess for determining the position within the lower part"
                                                               " of the image of the right rail", default=870)
    parser.add_argument('--watchHoughLowerRails', type=bool, default=False)
    parser.add_argument("--show", type=bool, help="Show output image flow", default=True)
    parser.add_argument('--seekframe',default=0,type=int,help="seek in frames")

    args = parser.parse_args()

    # ---------------------------------------------INITIALIZATION--------------------------------------------

    cap = ImageIOVideoCapture(args.input)
    if args.seekframe != 0:
        cap.seek(args.seekframe)

    firstFrame = True

    # frame ID
    count = 0

    # init template dimensions for railway tracking
    w = args.tmpWidth
    h = args.tmpHeight

    # image size
    sizeX = args.imgSizeX  # width
    sizeY = args.imgSizeY  # height

    # init expected coordinates for template lower left corner
    MAXl = []

    # initialization for line detection using Hough
    expt_startLeft = args.startLeftRailGuess
    expt_startRight = args.startRightRailGuess

    # initialize feature matcher
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


    # create railway detection object
    railway_objects = RailwayObjectDetection(ml_rail, cl_rail, mr_rail, cr_rail, foe)

    # initialization of lists to contain matched features across two consecutive frames
    kps_on_track = []
    kps_on_track_old = []

    kps_on_track_empty = 1

    # list to track dynamic Focus of Expansion displacement along x
    dynFOEx = []

    # init cascade classifier for (white) turn poles detection
    palb_cascade = cv2.CascadeClassifier(args.cascadeTurn)

    # init cascade classifier for kilometer sign detection
    km_cascade = cv2.CascadeClassifier(args.cascadeKm)

    # lists to contain matched features on detected objects
    pole_points = []
    pole_oldpoints = []

    km_points = []
    km_old_points = []

    left_edges = []
    right_edges = []

    left_xseq = []
    left_yseq = []
    right_xseq = []
    right_yseq = []

    # view patch with rail lines detected using HoughLines
    if args.watchHoughLowerRails == True:
        watch_rails = 1
    else:
        watch_rails = 0

    # view correlation result along x (1)
    watch_xcorr = 0

    # activate Lorentzian curve weighting (1)
    weighted_version = 1

    # Next frame availability
    r = True

    while(r == True):
        r, frame = cap.read()

        if frame is None:
            break

        # reset lists of matched point features for each processed frame
        pole_points[:] = []
        pole_oldpoints[:] = []

        # reset lists of matched point features on km signs for each processed frame
        km_points[:] = []
        km_old_points[:] = []

        matchImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # extract ORB features
        kp, des = orb.detectAndCompute(matchImage, None)

        # INIT HAAR CASCADE CLASSIFIERS
        poles_bbox = palb_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(16, 33),
                                                 flags=cv2.CASCADE_SCALE_IMAGE)

        km_bbox = km_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=3, minSize=(22, 107),
                                              flags=cv2.CASCADE_SCALE_IMAGE)

        if not firstFrame:

            # brute force feature matching
            matches = bf.match(des_old, des)
            matches = sorted(matches, key=lambda x: x.distance)

            # init array to store FOE measurements for each new frame
            del dynFOEx[:]

            if kps_on_track_empty == 0:
                del kps_on_track[:]

            # TEMPLATE MATCHING OF RAILS
            # init rail objects to start template-match detection
            railL = Rail(int(expt_startLeft) - 7, sizeY, w, h)
            railR = Rail(int(expt_startRight) - 7, sizeY, w, h)

            # handling of left and rigth rail lines detection within a portion of image,
            # expt_startLeft is the expected position of the lower edge of the line segment fitting the rail
            # expressed in full image coordinates
            eval_frameL, start_colL, end_colL, expt_startLeft = railL.extract_rail_line(frame,
                                                                                           start_rows=1000,
                                                                                           end_rows=np.size(
                                                                                               frame, 0),
                                                                                           start_cols=610,
                                                                                           end_cols=1000,
                                                                                           canny_min=40,
                                                                                           canny_max=140,
                                                                                           hough_I=45,
                                                                                           length=400,
                                                                                           theta_min=2.5,
                                                                                           theta_max=2.65,
                                                                                           expt_start=expt_startLeft,
                                                                                           max_displacement=15,
                                                                                           )

            eval_frameR, start_colR, end_colR, expt_startRight = railR.extract_rail_line(frame,
                                                                                            start_rows=1000,
                                                                                            end_rows=1080,
                                                                                            start_cols=600,
                                                                                            end_cols=950,
                                                                                            canny_min=40,
                                                                                            canny_max=200,
                                                                                            hough_I=40,
                                                                                            length=100,
                                                                                            theta_min=2.9,
                                                                                            theta_max=3.1,
                                                                                            expt_start=expt_startRight,
                                                                                            max_displacement=20,
                                                                                            )

            # view the portion of image where rail lines have been detected
            railL.patch_wRails(watch_rails, eval_frameL, eval_frameR)

            newFOEx_rail, newFOEy_rail, ml, cl, mr, cr, left_edges, right_edges = tmp_match_rail_lines(w, frame, matchImage, weighted_version, watch_xcorr, MAXl, railL, railR, left_edges, left_xseq, left_yseq, right_edges, right_xseq, right_yseq)

            #----------------------------------------POINT FEATURE MATCHING--------------------------------------------
            for k in range(0, len(matches)):

                q = matches[k].queryIdx  # old frame
                t = matches[k].trainIdx  # new frame

                # keypoints coordinates p1 in current frame
                x = kp[t].pt[0]
                y = kp[t].pt[1]

                # keypoints coordinates p2 in previous frame
                x_old = kp_old[q].pt[0]
                y_old = kp_old[q].pt[1]

                # CHECK PRESENCE OF TURN POLES
                # list focus of expansion coordinates, each computed as the intersection of the displacement of feature within
                # the same bounding box which contain a pole
                dynFOEx, pole_points, pole_oldpoints = railway_objects.check_for_turn(poles_bbox, poles_bb_old, x, y, x_old, y_old, frame, mask, dynFOEx, pole_points, pole_oldpoints)

                for (xv, yv, wi, he) in km_bbox:
                    cv2.rectangle(frame, (xv, yv), (xv + wi, yv + he), (0, 0, 255), 2)
                km_points, km_old_points = railway_objects.get_km_features(km_bbox, km_bb_old, x, y, x_old, y_old,
                                                                           km_points, km_old_points)

                # POINT FEATURES FILTERING ACCORDING TO THE OPTICAL FLOW
                filter_features_matching(foenew, x, x_old, y, y_old, mask)

            # Focus of Expansion estimate given by the intersection of the lines fitting the rails.
            # The Focus of Expansion coordinates are updated according to a threshold check
            thresh_foe = 25
            if abs(newFOEx_rail - foe[0]) > thresh_foe:

                cv2.line(frame, (newFOEx_rail, newFOEy_rail), (left_edges[0][0], left_edges[0][1]), (0, 0, 255), 1)
                cv2.line(frame, (newFOEx_rail, newFOEy_rail), (right_edges[0][0], right_edges[0][1]),
                         (0, 0, 255), 1)

                cv2.drawMarker(frame, (newFOEx_rail, newFOEy_rail), (0, 0, 255),
                               markerType=cv2.MARKER_CROSS,
                               markerSize=20,
                               thickness=2,
                               line_type=cv2.LINE_AA)

            else:
                newFOEx_rail = foe[0]

            # the previous value of foenew is updated whether turn poles are detected in the frame
            if len(poles_bbox) > 0 and len(dynFOEx) > 0:

                newFOEx_poles = np.mean(dynFOEx, dtype=int)

                # update of the FOE coordinates also determined by the rail lines when different from nominal value
                if abs(newFOEx_rail - foe[0]) > 0:
                    newFOEx = (newFOEx_rail + newFOEx_poles) / 2
                else:
                    newFOEx = newFOEx_poles

                foenew = np.array((int(newFOEx), foe[1]))

                # dynamic FOE (turn condition) is represented as a blue cross
                cv2.drawMarker(frame, (newFOEx_poles, foe[1]), (0, 255, 255),
                               markerType=cv2.MARKER_CROSS,
                               markerSize=20,
                               thickness=2,
                               line_type=cv2.LINE_AA)

            # when the difference between dynamically computed FOE under turn condition and nominal FOE is below a threshold
            # it is assumed to be under straight track condition
            elif abs(foenew[0] - foe[0]) < thresh_foe:
                foenew = foe

            if args.show and watch_rails == 0:
                # nominal FOE (straight condition) is represented as a green cross
                cv2.drawMarker(frame, (foe[0], foe[1]), (0, 255, 0),
                               markerType=cv2.MARKER_CROSS,
                               markerSize=20,
                               thickness=2,
                               line_type=cv2.LINE_AA)

                #---------------------------DRAW matched keypoints on frame--------------------------------------------

                frame = cv2.drawKeypoints(frame, kp, frame)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frameC = cv2.add(frame, mask)

                frame_small = cv2.resize(frameC, (int(0.9 *sizeX), int(0.9 * sizeY)))
                cv2.imshow('frame', frame_small)

        else:
            # when firstframe == 1, initialise dynamic position of FOE as nominal
            foenew = foe
            firstFrame = False

        old_frame = frame
        kp_old = kp
        des_old = des
        poles_bb_old = poles_bbox[:]
        km_bb_old = km_bbox[:]

        del kps_on_track_old[:]

        mask = np.zeros_like(old_frame)

        # update frame id
        count = count + 1


if __name__ == '__main__':
    main()