# -*- coding: utf-8 -*-
'''
 * @Author: KD Zhang
 * @Date: 2023-02-28 19:26:35  
'''
import cv2
import numpy as np
from skimage.filters import frangi
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
import scipy.ndimage as ndi
from icecream import ic
import imageio


def calc_gradient(image):
    jsobel = ndi.sobel(image, axis=1)  # along width axis
    isobel = ndi.sobel(image, axis=0)  # along height axis
    mag = np.hypot(isobel, jsobel)
    return mag, isobel, jsobel


def norm_coord(arr, limit):
    arr[arr < 0] = 0
    arr[arr >= limit] = limit - 1
    return arr


def moving_points(cx, cy, r, mag, roi):
    sampling_interval = 90
    sampling_angles = np.linspace(0, 2 * np.pi, sampling_interval)
    small_r = r
    big_r = r * 1.7 * 1.5
    steps = int(big_r) - small_r
    cand_mask = np.zeros_like(mag)
    radius = small_r
    prev_x = (cx + small_r * np.cos(sampling_angles) * (-1)).astype(int)
    prev_y = (cy + small_r * np.sin(sampling_angles)).astype(int)
    h, w = mag.shape
    for i in range(steps):
        radius += 1
        x_shift = radius * np.cos(sampling_angles) * (-1)  # shift along h
        y_shift = radius * np.sin(sampling_angles)  # shift along w
        new_x = (cx + x_shift).astype(int)
        new_y = (cy + y_shift).astype(int)
        new_x = norm_coord(new_x, h)
        new_y = norm_coord(new_y, w)
        prev_mag = mag[prev_x, prev_y]
        post_mag = mag[new_x, new_y]
        prev_roi = roi[prev_x, prev_y]
        post_roi = roi[new_x, new_y]
        cand_mask[new_x[post_mag > prev_mag], new_y[post_mag > prev_mag]] = 1
        # cand_mask[new_x[prev_roi < post_roi], new_y[prev_roi < post_roi]] = 0
        # h_pos = np.where(post_mag > prev_mag)[0]
        # cand_mask[prev_x[h_pos], prev_y[h_pos]] = 0
        prev_x = new_x
        prev_y = new_y
    return cand_mask


def contour_refine_func(blur_ct_array, ct_array, cx, cy, radius, shrink, dilate, percent, vis=False, circle_peaks=5):
    # Heissian filter to the clean CT image
    h, w = ct_array.shape
    ct_array_for_processing = ct_array.astype(np.float)
    max_ca = np.max(ct_array_for_processing)
    scale_factor = 255 / max_ca
    ct_array_for_processing *= scale_factor
    p_mask = frangi(ct_array_for_processing)
    enhanced_ct_array = (1 - p_mask) * blur_ct_array
    bone_thres = np.percentile(enhanced_ct_array, percent)
    bone_mask = enhanced_ct_array > bone_thres
    mag, isobel, jsobel = calc_gradient(enhanced_ct_array)
    cand_mask = moving_points(cx, cy, radius, mag, enhanced_ct_array)
    cand_mask = cand_mask * bone_mask

    enhanced_ct_array_vis = np.clip(enhanced_ct_array, a_min=-125, a_max=275)
    enhanced_ct_array_vis = ((enhanced_ct_array_vis + 127) / (275 + 125)) * 255.
    # np.save('enhanced_ct_array_vis.npy', enhanced_ct_array_vis)
    enhanced_ct_array_vis_r = enhanced_ct_array_vis.copy()
    enhanced_ct_array_vis_g = enhanced_ct_array_vis.copy()
    enhanced_ct_array_vis_b = enhanced_ct_array_vis.copy()
    enhanced_ct_array_vis_r[cand_mask == 1] = 249
    enhanced_ct_array_vis_g[cand_mask == 1] = 78
    enhanced_ct_array_vis_b[cand_mask == 1] = 1
    enhanced_ct_array_vis = np.stack([enhanced_ct_array_vis_r, enhanced_ct_array_vis_g, enhanced_ct_array_vis_b], axis=-1)
    imageio.imwrite('/home/zkd/newJobs/bone_seg/graph_cut/cand_mask_vis_noFG_noNMS_v2.png', enhanced_ct_array_vis)
    temp = input('kkpsa')

    h_pos, w_pos = np.where(cand_mask == 1)
    try_radii = np.arange(radius, radius * shrink * dilate, 2)
    hough_res = hough_circle(cand_mask, try_radii)
    accums, center_w, center_h, radii = hough_circle_peaks(hough_res, try_radii, total_num_peaks=circle_peaks)
    best_idx = np.argmin(np.abs(radii - radius * shrink))
    ch, cw, cr = center_h[best_idx], center_w[best_idx], int(radii[best_idx])
    rr, cc = circle_perimeter(ch, cw, cr, shape=(h, w))

    # statistical filtering with the detected hough circle
    v2 = np.abs((h_pos - ch) ** 2 + (w_pos - cw) ** 2 - cr ** 2)
    v = np.sqrt(v2)
    s = np.sqrt(np.sum(v2 / len(h_pos) - 1))
    error_points = v > s
    cand_mask[h_pos[error_points], w_pos[error_points]] = 0

    if vis:
        scaled_image = (np.maximum(enhanced_ct_array, 0) / enhanced_ct_array.max()) * 255.0
        scaled_image2 = (np.maximum(enhanced_ct_array, 0) / enhanced_ct_array.max()) * 255.0
        h_pos, w_pos = np.where(cand_mask == 1)
        scaled_image[h_pos, w_pos] = 255
        cv2.imwrite('/home/zkd/newJobs/bone_seg/graph_cut/debug_rets/refined_contour_right.png', scaled_image)
        scaled_image2[rr, cc] = 255
        cv2.imwrite('/home/zkd/newJobs/bone_seg/graph_cut/debug_rets/refined_contour_right_hough.png', scaled_image2)
    return cand_mask, ch, cw, cr
