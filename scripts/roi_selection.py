# -*- coding: utf-8 -*-
'''
 * @Author: KD Zhang
 * @Date: 2023-02-27 14:13:35  
'''
import numpy as np
from skimage.measure import label
import numpy as np


def side_processing(side, thres, use_column, use_label, pivot, axial_selection):
    # left side processing
    side[side < thres] = 0
    side[side >= thres] = 1
    if use_column > 0:
        if use_label:
            labeled_mask, num = label(side, background=0, return_num=True)
            max_connection = 0
            for i in range(1, num + 1):
                region_mask = (labeled_mask == i)
                mask_sum = np.sum(region_mask)
                if mask_sum > max_connection:
                    max_connection = i
            region_mask = (labeled_mask == max_connection)
            h_pos, w_pos = np.where(region_mask == 1)
            max_w_pos, min_w_pos = np.max(w_pos), np.min(w_pos)
            target_column = (max_w_pos + min_w_pos) // 2
        else:
            target_column = pivot // 2
        # initialize a mask, with the target column 1, other columns 0
        mask = np.zeros_like(side)
        mask[:, target_column] = 1
    else:
        mask = np.ones_like(side)
    h_pos, w_pos = np.where(side * mask == 1)
    target_axial = (np.max(h_pos) + np.min(h_pos)) // 2
    roi_mask = np.zeros_like(side)
    roi_mask[target_axial - axial_selection // 2: target_axial + axial_selection // 2, :] = 1
    up = target_axial - axial_selection // 2
    bottom = target_axial + axial_selection // 2
    return roi_mask, up, bottom


def side_processing_preSeg(seg_side, use_column, use_label, pivot, axial_selection):
    if use_column > 0:
        if use_label:
            labeled_mask, num = label(seg_side, background=0, return_num=True)
            max_connection = 0
            for i in range(1, num + 1):
                region_mask = (labeled_mask == i)
                mask_sum = np.sum(region_mask)
                if mask_sum > max_connection:
                    max_connection = i
            region_mask = (labeled_mask == max_connection)
            h_pos, w_pos = np.where(region_mask == 1)
            max_w_pos, min_w_pos = np.max(w_pos), np.min(w_pos)
            target_column = (max_w_pos + min_w_pos) // 2
        else:
            target_column = pivot // 2
        mask = np.zeros_like(seg_side)
        mask[:, target_column] = 1
    else:
        mask = np.ones_like(seg_side)
    h_pos, w_pos = np.where(seg_side * mask == 1)
    target_axial = (np.max(h_pos) + np.min(h_pos)) // 2
    roi_mask = np.zeros_like(seg_side)
    roi_mask[target_axial - axial_selection // 2: target_axial + axial_selection // 2, :] = 1
    up = target_axial - axial_selection // 2
    bottom = target_axial + axial_selection // 2
    return roi_mask, up, bottom


def roi_selection(ct_array, thres, use_column, use_label):
    pivot = ct_array.shape[1] // 2
    axial_selection = ct_array.shape[0] // 2
    left_side = ct_array[:, :pivot]
    right_side = ct_array[:, pivot:]

    assert thres > 0, 'Threshold must be larger than 0!'

    left_roi_mask, left_up, left_bottom = side_processing(left_side, thres, use_column, use_label, pivot,
                                                          axial_selection)
    right_roi_mask, right_up, right_bottom = side_processing(right_side, thres, use_column, use_label, pivot,
                                                             axial_selection)
    return left_roi_mask, left_up, left_bottom, right_roi_mask, right_up, right_bottom


def roi_selection_preSegs(pre_seg, use_column, use_label):
    pivot = pre_seg.shape[1] // 2
    axial_selection = pre_seg.shape[0] // 2
    left_preSeg = pre_seg[:, :pivot]
    right_preSeg = pre_seg[:, pivot:]

    left_roi_mask, left_up, left_bottom = side_processing_preSeg(left_preSeg, use_column, use_label, pivot,
                                                                 axial_selection)
    right_roi_mask, right_up, right_bottom = side_processing_preSeg(right_preSeg, use_column, use_label, pivot,
                                                                    axial_selection)
    return left_roi_mask, left_up, left_bottom, right_roi_mask, right_up, right_bottom


def roi_selection_preSegs_individual(pre_seg, use_column, use_label):
    pivot = 0
    axial_selection = pre_seg.shape[0] // 2
    roi_mask, roi_up, roi_bottom = side_processing_preSeg(pre_seg, use_column, use_label, pivot, axial_selection)
    return roi_up, roi_bottom
