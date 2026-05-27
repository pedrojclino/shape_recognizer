import numpy as np
import cv2


def rotate_vector(vector, theta, offset):
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([
        [c, -s],
        [s,  c]
    ])
    
    return (rotation_matrix @ (vector-offset).T ).T + offset

def center_poly(a_pts, b_pts):

    len_poly = len(b_pts)
    if len_poly == 0 or len(a_pts) == 0:
        return b_pts

    return b_pts - b_pts[0,:] + a_pts[0,:]

def align_poly(a_pts, b_pts):

    len_poly = len(a_pts)
    if len_poly <= 1 or len_poly != len(b_pts):
        return b_pts

    # Find the angle which aligns the first edge
    a_edge = a_pts[1,:] - a_pts[0,:]
    b_edge = b_pts[1,:] - b_pts[0,:]
    a_angle = np.atan2(a_edge[1],a_edge[0])
    b_angle = np.atan2(b_edge[1],b_edge[0])
    angle = b_angle - a_angle
    p_origin = a_pts[0,:]

    # Propagate to all the other edges 
    return np.apply_along_axis(lambda row : rotate_vector(row, -angle, p_origin), 1, b_pts)

def scale_poly(a_pts, b_pts):

    contour = a_pts.astype(np.float32).reshape(-1, 1, 2)
    a_perimeter = cv2.arcLength(contour, closed=True)
    contour = b_pts.astype(np.float32).reshape(-1, 1, 2)
    b_perimeter = cv2.arcLength(contour, closed=True)

    f = a_perimeter / b_perimeter
    p_origin = b_pts[0,:]
    return f * (b_pts - p_origin) + p_origin

def compare_polys(a_pts, b_pts):

    len_poly = len(a_pts)
    if len_poly != len(b_pts):
        return 0
    
    contour = a_pts.astype(np.float32).reshape(-1, 1, 2)
    a_perimeter = cv2.arcLength(contour, closed=True)
    contour = b_pts.astype(np.float32).reshape(-1, 1, 2)
    b_perimeter = cv2.arcLength(contour, closed=True)

    diff_length = 0
    for i in range(0, len_poly-1):
        a_edge = a_pts[i+1,:] - a_pts[i,:]
        b_edge = b_pts[i+1,:] - b_pts[i,:]

        v_diff = (a_edge / a_perimeter) - (b_edge / b_perimeter)
        diff_length += np.linalg.norm(v_diff)

    return 1 - diff_length / 2

def float_range(start, stop, step):
    while start < stop:
        yield start
        start += step

def grow_poly(contour, target_length, start_sensitivity):

    # Find more edges until equal
    for sensitivity in float_range(start_sensitivity, 0.8, 0.02):
        perimeter = cv2.arcLength(contour, closed = True)
        contour_poly = cv2.approxPolyDP(contour, epsilon = sensitivity * perimeter, closed = True)

        if len(contour_poly) == target_length:
            break

    return contour_poly

def gen_same_size_poly(a_contour, b_contour):

    sensitivity = 0.02

    perimeter = cv2.arcLength(a_contour, closed = True)
    a_contour_poly = cv2.approxPolyDP(a_contour, epsilon = sensitivity * perimeter, closed = True)
    len_a_contour_poly = len(a_contour_poly)

    perimeter = cv2.arcLength(b_contour, closed = True)
    b_contour_poly = cv2.approxPolyDP(b_contour, epsilon = sensitivity * perimeter, closed = True)
    len_b_contour_poly = len(b_contour_poly)

    if len_a_contour_poly < len_b_contour_poly:
        a_contour_poly = grow_poly(a_contour, len_b_contour_poly, sensitivity)
    elif len_b_contour_poly < len_a_contour_poly:
        b_contour_poly = grow_poly(b_contour, len_a_contour_poly, sensitivity)

    success = bool(len(b_contour_poly) == len(a_contour_poly))

    return success, a_contour_poly.squeeze(), b_contour_poly.squeeze()

def compare_contours(a_contour, b_contour):

    max_score = 0
    best_contour_poly = np.empty((0,2))

    success, a_contour_poly, b_contour_poly = gen_same_size_poly(a_contour, b_contour)
    if success:
        # Compare on the best indexed edge basis
        for i in range(0,len(b_contour_poly)):
            contour_poly = np.roll(b_contour_poly, shift = i, axis=0)
            contour_poly = center_poly(a_contour_poly, contour_poly)
            contour_poly = align_poly(a_contour_poly, contour_poly)
            contour_poly = scale_poly(a_contour_poly, contour_poly)
            score = compare_polys(a_contour_poly, contour_poly)

            if max_score < score:
                max_score = score
                best_contour_poly = contour_poly

    return max_score, a_contour_poly, best_contour_poly

def classify_descriptor(unknown_contour, labeled_img_datas):

    classification = dict()
    for label, img_datas in labeled_img_datas.items():

        # Compare against all the descriptors on the model
        best_score = 0
        best_known_poly_aprox = np.empty((0,2))
        best_descriptor_poly = np.empty((0,2))
        for img_data in img_datas:
            for descriptor in img_data.descriptors:

                # Compare the descriptor contours
                score, descriptor_poly, known_poly_aprox = compare_contours(unknown_contour, descriptor.contour)

                # Save the best correspondance
                if best_score < score:
                    best_score = score
                    best_known_poly_aprox = known_poly_aprox
                    best_descriptor_poly = descriptor_poly

        # Set for each label the 'score' of the unknown_contour to be be identical to a known polygon shape
        classification[label] = {"score" : best_score, "input_poly": best_descriptor_poly, "poly" : best_known_poly_aprox}

    return classification