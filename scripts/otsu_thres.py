# -*- coding: utf-8 -*-
'''
 * @Author: KD Zhang
 * @Date: 2023-02-27 16:43:35  
'''
from skimage.filters import threshold_otsu
import numpy as np
from skimage import morphology
from skimage.measure import label
from scipy.ndimage import binary_fill_holes


def find_largest_connection(binary):
    labeled_mask, num = label(binary, background=0, return_num=True)
    max_label, max_area = 0, 0
    for i in range(1, num + 1):
        region_mask = (labeled_mask == i)
        area = np.sum(region_mask)
        if area > max_area:
            max_label = i
            max_area = area
    assert max_label > 0
    return labeled_mask == max_label


def post_process(binary):
    width = 20
    binary = morphology.remove_small_holes(binary, area_threshold=width ** 2)
    binary = binary_fill_holes(binary)
    # binary = find_largest_connection(binary)
    return binary


def otsu_seg(ct_array, up, bottom):
    roi_area = ct_array[up:bottom, :]
    roi_normed = roi_area.copy()
    roi_normed[roi_normed < 0] = 0
    thres = threshold_otsu(roi_normed)
    binary = roi_area > thres
    binary = post_process(binary)
    return binary
