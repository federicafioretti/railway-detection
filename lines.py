
# considering a line described as m*x + c = y:

# compute line polynomial coefficient from two given points in a single list
def points2coeffs(points):
    p1lx = points[0][0]
    p1ly = points[0][1]
    p2lx = points[1][0]
    p2ly = points[1][1]
    if p1lx != p2lx:
        ml = float(p2ly - p1ly) / (p2lx - p1lx)
    else:
        ml = 0
    cl = float(p1ly - ml * (p1lx))
    return ml, cl

# compute intersection point within image between a line having coefficients (m, c)
# with the horizon (given by the y-coordinate of the principal point, 631 from calibration)
def lines2intersection_hor(m, c, Y_i=631):
    X_i = int((Y_i - c) / (m + 0.0001))                 # avoid division by zero
    return X_i, Y_i

# given two vectors m, c including the polynomial coefficients from different lines
# returns their intersection
def lines2intersection(m, c):
    Y_i = int((m[0] * c[1] - m[1] * c[0]) / (m[0] - m[1]))
    X_i = int((Y_i - c[0]) / m[0])
    return X_i, Y_i