# -*- coding: utf-8 -*-
'''
 * @Author: KD Zhang
 * @Date: 2023-02-28 19:26:35  
'''
import numpy as np
from skimage.segmentation import active_contour
from skimage.draw import circle_perimeter, polygon2mask
from skimage.transform import hough_circle, hough_circle_peaks
from scipy.ndimage import binary_fill_holes
from icecream import ic

H, W = 512, 512


def map_back_new(cand_mask, ch, up, bottom):
    new_ch = up + ch
    new_mask = np.zeros((H, W // 2))
    new_mask[up:bottom, :] = cand_mask
    return new_mask, new_ch


def map_counter_new(up, bottom, mask, ch):
    new_ch = ch - up
    new_mask = mask[up:bottom, :]
    return new_mask, new_ch


def determine_center_plane_coordinate_new(chs, cws, indices, ups):
    assert len(chs) == len(cws) == len(indices)
    new_chs = []
    for i in range(len(chs)):
        idx = indices[i]
        left_up = ups[idx]
        ch = chs[idx]
        ch = ch + left_up
        new_chs.append(ch)

    # Count the mode by combining the considering of x and y axis
    coords_stats = {}
    max_key, max_occurs = None, 0
    for i in range(len(new_chs)):
        key = f'{new_chs[i]}_{cws[i]}'
        if not key in coords_stats:
            coords_stats[key] = 1
            if max_key is None:
                max_key = key
                max_occurs = 1
        else:
            coords_stats[key] += 1
            if coords_stats[key] > max_occurs:
                max_occurs = coords_stats[key]
                max_key = key
    center_ch, center_cw = int(max_key.split('_')[0]), int(max_key.split('_')[1])

    return center_ch, center_cw


def determine_center_axial_and_radius_vote(agg_crs, agg_idx):
    crs_max = np.max(agg_crs)
    selected_idx = np.array(agg_idx)[agg_crs == crs_max]
    mean_idx = int(np.mean(selected_idx))
    Z0 = mean_idx
    R = crs_max  # transform between space to axial coordinate
    return Z0, R


def generate_femoral_head_masks(center_ch, center_cw, Z0, R, space_x, space_z, interval_slices, mask_shape):
    femoral_head_mask_sequences = []
    slice_idx = np.linspace(Z0 - interval_slices + 1, Z0 + interval_slices - 1, 2 * interval_slices - 1)
    theta = np.arcsin(np.abs(slice_idx - Z0) / (R * space_x / space_z))
    radius = R * np.cos(theta)
    for i in range(radius.shape[0]):
        rad = int(radius[i])
        rr, cc = circle_perimeter(center_ch, center_cw, rad)
        circle_coordinates = np.stack([rr, cc], axis=1)
        filled_circle = binary_fill_holes(polygon2mask(mask_shape, circle_coordinates))
        femoral_head_mask_sequences.append(filled_circle.astype(np.int32))
    return femoral_head_mask_sequences, radius


def snake_filter(snake, h, w):
    pointer = snake.shape[0]
    for i in range(snake.shape[0]):
        if snake[i, 0] >= h or snake[i, 1] >= w or snake[i, 0] < 0 or snake[i, 1] < 0:
            snake[i, :] = snake[pointer - 1]
            pointer -= 1
    return snake[:pointer, :]


def active_contour_propagation(cand_mask, ch, cw, cr, space_x, space_z, ups, bottoms, cts, blur_cts, pivot, down_scale=0.8, up_scale=1.2,
                               mask_shape=(512, 256)):
    h_pos, w_pos = np.where(cand_mask == 1)
    initial_snake = np.stack((h_pos, w_pos), axis=1)

    h, w = cand_mask.shape

    z_cr = int(cr * space_x / space_z)
    assert len(cts) == 2 * z_cr + 1
    rr, cc = circle_perimeter(ch, cw, cr, shape=cand_mask.shape)
    new_cand_mask = np.zeros_like(cand_mask)
    new_cand_mask[rr, cc] = 1
    new_mask, new_ch = map_back_new(new_cand_mask, ch, ups[pivot], bottoms[pivot])

    # generate initial snake
    down_snakes = [initial_snake]

    # center height, center width and radius of each circle
    down_chs, down_cws, down_crs = [ch], [cw], [cr]
    down_idx = []

    # down iterative processing
    counter = 0
    for i in range(pivot + 1, min(len(cts), pivot + z_cr + 1)):
        down_idx.append(i)
        new_blur_ct = blur_cts[i]
        new_up, new_bottom = ups[i], bottoms[i]
        map_mask, map_ch = map_counter_new(new_up, new_bottom, new_mask, new_ch)

        scaled_image = (np.maximum(new_blur_ct, 0) / new_blur_ct.max()) * 255.0
        h_pos, w_pos = np.where(map_mask == 1)
        snake = np.stack((h_pos, w_pos), axis=1)
        snake = active_contour(scaled_image, snake)
        snake = snake.astype(np.int)

        snake = snake_filter(snake, h, w)

        # Hough transform to fit a circle of current femoral head
        try_radii = np.arange(int(down_crs[counter] * down_scale), int(down_crs[counter] * up_scale))
        hough_mask = np.zeros_like(cand_mask)
        hough_mask[snake[:, 0], snake[:, 1]] = 1
        hough_res = hough_circle(hough_mask, try_radii)
        accum, center_w, center_h, radii = hough_circle_peaks(hough_res, try_radii, total_num_peaks=1)
        # ic(accum, center_h, center_w, radii)
        next_ch, next_cw, next_cr = center_h[0], center_w[0], radii[0]

        down_chs.append(next_ch)
        down_cws.append(next_cw)
        down_crs.append(next_cr)
        counter += 1

        rr, cc = circle_perimeter(next_ch, next_cw, next_cr, shape=cand_mask.shape)
        snake = np.stack((rr, cc), axis=1)

        down_snakes.append(snake)

    up_snakes = [initial_snake]
    up_idx = []
    up_chs, up_cws, up_crs = [ch], [cw], [cr]
    counter = 0
    for i in range(pivot - 1, max(0, pivot - z_cr) - 1, -1):
        up_idx.append(i)
        new_blur_ct = blur_cts[i]
        new_up, new_bottom = ups[i], bottoms[i]
        map_mask, map_ch = map_counter_new(new_up, new_bottom, new_mask, new_ch)
        scaled_image = (np.maximum(new_blur_ct, 0) / new_blur_ct.max()) * 255.0

        h_pos, w_pos = np.where(map_mask == 1)

        snake = np.stack((h_pos, w_pos), axis=1)
        snake = active_contour(scaled_image, snake)
        snake = snake.astype(np.int)

        snake = snake_filter(snake, h, w)

        # Hough transform to fit a circle of current femoral head
        try_radii = np.arange(int(up_crs[counter] * down_scale), int(up_crs[counter] * up_scale))
        hough_mask = np.zeros_like(cand_mask)
        hough_mask[snake[:, 0], snake[:, 1]] = 1

        hough_res = hough_circle(hough_mask, try_radii)
        accum, center_w, center_h, radii = hough_circle_peaks(hough_res, try_radii, total_num_peaks=1)
        next_ch, next_cw, next_cr = center_h[0], center_w[0], radii[0]

        up_chs.append(next_ch)
        up_cws.append(next_cw)
        up_crs.append(next_cr)
        counter += 1

        rr, cc = circle_perimeter(next_ch, next_cw, next_cr, shape=cand_mask.shape)
        snake = np.stack((rr, cc), axis=1)

        up_snakes.append(snake)

    # determine center plane coordinate
    agg_chs = up_chs[::-1] + down_chs[1:]
    agg_cws = up_cws[::-1] + down_cws[1:]
    agg_idx = up_idx[::-1] + [pivot] + down_idx
    agg_crs = up_crs[::-1] + down_crs[1:]

    center_ch, center_cw = determine_center_plane_coordinate_new(agg_chs, agg_cws, agg_idx, ups)
    Z0, R = determine_center_axial_and_radius_vote(agg_crs, agg_idx)
    interval_slices = int(R * space_x / space_z)  # The truly radius of the femoral head
    femoral_head_mask_sequences, radius = generate_femoral_head_masks(center_ch, center_cw, Z0, R, space_x, space_z, interval_slices,
                                                                      mask_shape)
    assert len(femoral_head_mask_sequences) == 2 * interval_slices - 1
    return femoral_head_mask_sequences, center_ch, center_cw, radius
