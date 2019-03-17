import cv2
import numpy as np
from lines import lines2intersection_hor

def tmp_match_rail_lines(w, frame, matchImage, weighted_version, watch_xcorr, MAXl, railL, railR, left_edges, left_xseq, left_yseq, right_edges, right_xseq, right_yseq):

    # draw first template
    railL.mark(frame, 0)
    railR.mark(frame, 0)

    # init expected coordinates for template lower left corner, where to look for the upper template containing
    # a left rail element
    Ml = (2 * w, matchImage.shape[0])

    # init expected coordinates for template lower left corner, where to look for the upper template containing
    # a right rail element
    Mr = (2 * w, matchImage.shape[0])
    MAXl.append(Ml[0])

    # init lists to contain the sequence of template corners, fitting left and right rail
    left_edges[:] = []
    right_edges[:] = []

    left_xseq[:] = []
    left_yseq[:] = []
    right_xseq[:] = []
    right_yseq[:] = []

    # template matching of rails, using 30 templates, one located at y 30 pixel less than the other
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

        # line fitting the rails between the 1st and the 5th templates
        if loop == 1 or loop == 5:
            left_edges.append((xl, yl))
            right_edges.append((xr, yr))

        # polyfit of the first 15 templates
        if loop < 16:
            left_xseq.append(xl)
            left_yseq.append(yl)
            right_xseq.append(xr)
            right_yseq.append(yr)

        cv2.drawMarker(frame, (xl, yl), (0, 0, 255),
                       markerType=cv2.MARKER_CROSS,
                       markerSize=5,
                       thickness=2,
                       line_type=cv2.LINE_AA)

        cv2.drawMarker(frame, (xr, yr), (0, 0, 255),
                       markerType=cv2.MARKER_CROSS,
                       markerSize=5,
                       thickness=2,
                       line_type=cv2.LINE_AA)

        loop = loop + 1

    # RAIL LINES EXTRACTION
    # compute coefficients of lines, fitting left and right rails
    left_rail_line = np.polyfit(left_xseq, left_yseq, 1)
    right_rail_line = np.polyfit(right_xseq, right_yseq, 1)

    # extract lines coefficients
    ml = left_rail_line[0]
    cl = left_rail_line[1]
    mr = right_rail_line[0]
    cr = right_rail_line[1]

    # compute intersection of a rail line and the horizon
    newFOEx_rail, newFOEy_rail = lines2intersection_hor(ml, cl)

    return newFOEx_rail, newFOEy_rail, ml, cl, mr, cr, left_edges, right_edges