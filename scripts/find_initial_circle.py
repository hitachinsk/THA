# -*- coding: utf-8 -*-
'''
 * @Author: KD Zhang
 * @Date: 2023-02-27 21:10:35  
'''
import cv2
import numpy as np
from skimage.measure import label
from scipy.spatial import ConvexHull
from scipy.spatial import distance
from skimage.draw import polygon, circle_perimeter


def define_circle(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return (None, np.inf)

    # Center of circle
    cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0]) ** 2 + (cy - p1[1]) ** 2)
    cx, cy, radius = int(cx), int(cy), int(radius)
    return ((cx, cy), radius)


# shade color to target point(s)
def shading(mask, cx, cy, color, bar=5):
    assert isinstance(color, tuple) or isinstance(color, list) and len(color) == 3
    if isinstance(cx, int) and isinstance(cy, int):
        mask[cx - bar:cx + bar, cy - bar:cy + bar, 0] = color[0]
        mask[cx - bar:cx + bar, cy - bar:cy + bar, 1] = color[1]
        mask[cx - bar:cx + bar, cy - bar:cy + bar, 2] = color[2]
    else:
        mask[cx, cy, 0] = color[0]
        mask[cx, cy, 1] = color[1]
        mask[cx, cy, 2] = color[2]
    return mask


# Only support the left side, right side can be inverted horizontally
def find_initial_circle_func(initial_plane, shrink=1.7, vis=False):
    labeled_mask, num = label(initial_plane, background=0, return_num=True)
    for i in range(1, num + 1):
        region_mask = (labeled_mask == i)
        area = np.sum(region_mask)
        h_pos, w_pos = np.where(region_mask == 1)
        if area < 300 and abs(initial_plane.shape[1] - w_pos.max()) < 20:
            # delete this connected field
            initial_plane[region_mask == 1] = 0
    # convex hull
    h_pos, w_pos = np.where(initial_plane)
    points = np.stack((h_pos, w_pos), axis=1)
    hull = ConvexHull(points)

    # get centroid
    cx = int(np.mean(hull.points[hull.vertices, 0]))  # h index of centroid
    cy = int(np.mean(hull.points[hull.vertices, 1]))  # w index of centroid

    h_vertices, w_vertices = points[hull.vertices, 0], points[hull.vertices, 1]
    rr, cc = polygon(h_vertices, w_vertices, initial_plane.shape)
    h, w = initial_plane.shape

    # up ray
    rr_indices = np.where(cc == cy)[0]
    up_point = (int(np.min(rr[rr_indices])), int(cy))  # intersection

    # right ray
    cc_indices = np.where(rr == cx)[0]
    right_point = np.max(cc[cc_indices])
    right_point = (int(cx), int(cy + (right_point - cy) // 2))

    # find the most distant point to the centroid
    valid_w = np.where(cc < cy)[0]
    valid_h = np.where(rr > cx)[0]
    valid_idx = np.intersect1d(valid_h, valid_w)

    XA = np.array([[cx, cy]])
    XB = np.stack([rr[valid_idx], cc[valid_idx]], axis=1)
    dist = distance.cdist(XA, XB, metric='cityblock')
    target_idx = np.argmax(dist, axis=1)
    target_rr, target_cc = list(XB[target_idx, :][0])

    # Point from great trochanter
    assert target_rr - cx > 0 and cy - target_cc > 0
    gt_point = (int(cx + (target_rr - cx) // 2), int(cy - (cy - target_cc) // 2))

    # draw a intiial circle based on these three feature points
    center, radius = define_circle(up_point, right_point, gt_point)
    radius = int(radius / shrink)
    # It has been normalized to small circle
    circle_rr, circle_cc = circle_perimeter(center[0], center[1], radius, shape=(h, w))

    # np.save('circle_rr.npy', circle_rr)
    # np.save('circle_cc.npy', circle_cc)

    # visualization
    if vis:
        mask = np.zeros((h, w, 3))
        # polygon
        mask = shading(mask, rr, cc, (0, 0, 255))
        # mask
        mask = shading(mask, h_pos, w_pos, (255, 255, 255))
        # centroid
        mask = shading(mask, cx, cy, (255, 0, 0))
        # up point
        mask = shading(mask, up_point[0], up_point[1], (255, 0, 255))
        # right point
        mask = shading(mask, right_point[0], right_point[1], (255, 0, 255))
        # gt point
        mask = shading(mask, gt_point[0], gt_point[1], (255, 0, 255))
        # circle
        mask = shading(mask, circle_rr, circle_cc, (200, 0, 200))
        cv2.imwrite('/home/zkd/newJobs/bone_seg/graph_cut/debug_rets/convex_mask_test_right.png', mask)

    return circle_rr, circle_cc, center[0], center[1], radius
