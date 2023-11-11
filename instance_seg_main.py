# -*- coding: utf-8 -*-
'''
 * @Author: KD Zhang
 * @Date: 2023-03-05 14:30:35  
'''
import os
import cv2
import time
import glob
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import gaussian_filter
from scripts.roi_selection import roi_selection, roi_selection_preSegs
from scripts.otsu_thres import otsu_seg
from scripts.find_initial_circle import find_initial_circle_func
from scripts.counter_refine import contour_refine_func
from scripts.active_counter_iteration import active_contour_propagation
import argparse


def read_masks(pre_seg):
    pre_seg_masks = sorted(glob.glob(os.path.join(pre_seg, '*.png')))
    pre_segs = []
    for i in range(len(pre_seg_masks)):
        pre_seg_mask = cv2.imread(pre_seg_masks[i])
        if len(pre_seg_mask.shape) == 3:
            pre_seg_mask = pre_seg_mask[:, :, 0]
        pre_seg_mask[pre_seg_mask > 0] = 1
        pre_segs.append(pre_seg_mask)
    return pre_segs


def rois_selection(series, roi_thres, use_column, use_label):
    n = series.shape[0]
    left_rois, left_ups, left_bottoms = [], [], []
    right_rois, right_ups, right_bottoms = [], [], []
    for i in range(n):
        sample = series[i]
        left_roi, left_up, left_bottom, right_roi, right_up, right_bottom = roi_selection(sample, roi_thres, use_column,
                                                                                          use_label)
        left_rois.append(left_roi)
        left_ups.append(left_up)
        left_bottoms.append(left_bottom)
        right_rois.append(right_roi)
        right_ups.append(right_up)
        right_bottoms.append(right_bottom)
    return left_rois, left_ups, left_bottoms, right_rois, right_ups, right_bottoms


def rois_selection_preSegs(pre_segs, use_column, use_label):
    left_rois, left_ups, left_bottoms = [], [], []
    right_rois, right_ups, right_bottoms = [], [], []
    for i in range(len(pre_segs)):
        pre_seg = pre_segs[i]
        left_roi, left_up, left_bottom, right_roi, right_up, right_bottom = roi_selection_preSegs(pre_seg, use_column,
                                                                                                  use_label)
        left_rois.append(left_roi)
        left_ups.append(left_up)
        left_bottoms.append(left_bottom)
        right_rois.append(right_roi)
        right_ups.append(right_up)
        right_bottoms.append(right_bottom)
    return left_rois, left_ups, left_bottoms, right_rois, right_ups, right_bottoms


def calc_pre_seg_func(ct_series, left_ups, left_bottoms, right_ups, right_bottoms):
    n, h, w = ct_series.shape
    assert len(left_ups) == n
    assert len(left_bottoms) == n
    assert len(right_ups) == n
    assert len(right_bottoms) == n
    left_binaries, right_binaries = [], []
    for i in range(n):
        ct = ct_series[i]
        left_side, right_side = ct[:, 0:w // 2], ct[:, w // 2:]
        left_binary = otsu_seg(left_side, left_ups[i], left_bottoms[i])
        right_binary = otsu_seg(right_side, right_ups[i], right_bottoms[i])
        left_binaries.append(left_binary)
        right_binaries.append(right_binary)
    return left_binaries, right_binaries


def proj_segs(segs, left_ups, left_bottoms, right_ups, right_bottoms):
    series_len = len(segs)
    h, w = segs[0].shape
    assert len(left_ups) == series_len
    assert len(left_bottoms) == series_len
    assert len(right_ups) == series_len
    assert len(right_bottoms) == series_len
    left_binaries, right_binaries = [], []
    for i in range(series_len):
        left_binary = segs[i][left_ups[i]:left_bottoms[i], 0:w // 2]
        right_binary = segs[i][right_ups[i]:right_bottoms[i], w // 2:]
        left_binaries.append(left_binary)
        right_binaries.append(right_binary)
    return left_binaries, right_binaries


def contour_prop(contour, ch, cw, cr, space_x, space_z, ups, bottoms, series, blurred_ct, initial_plane_idx, down_scale, up_scale,
                 mask_shape, pros_type='left'):
    w = blurred_ct.shape[2]
    z_cr = int(cr * space_x / space_z)
    blurred_ct_samples = blurred_ct[initial_plane_idx - z_cr:initial_plane_idx + z_cr + 1]
    ct_samples = series[initial_plane_idx - z_cr:initial_plane_idx + z_cr + 1]
    new_ups = ups[initial_plane_idx - z_cr:initial_plane_idx + z_cr + 1]
    new_bottoms = bottoms[initial_plane_idx - z_cr:initial_plane_idx + z_cr + 1]
    pivot = z_cr
    cropped_blurred_ct_samples = []
    cropped_ct_samples = []
    for i in range(len(ct_samples)):
        if pros_type == 'left':
            cropped_blurred_ct_sample = blurred_ct_samples[i, new_ups[i]:new_bottoms[i], 0:w // 2]
            cropped_ct_sample = ct_samples[i, new_ups[i]:new_bottoms[i], 0:w // 2]
        else:
            cropped_blurred_ct_sample = blurred_ct_samples[i, new_ups[i]:new_bottoms[i], w // 2:]
            cropped_ct_sample = ct_samples[i, new_ups[i]:new_bottoms[i], w // 2:]
            cropped_blurred_ct_sample = np.flip(cropped_blurred_ct_sample, axis=1)
            cropped_ct_sample = np.flip(cropped_ct_sample, axis=1)
        cropped_blurred_ct_samples.append(cropped_blurred_ct_sample)
        cropped_ct_samples.append(cropped_ct_sample)

    femoral_head_mask_sequences, ch, cw, radius = active_contour_propagation(contour, ch, cw, cr, space_x, space_z, new_ups, new_bottoms,
                                                                             cropped_ct_samples,
                                                                             cropped_blurred_ct_samples, pivot,
                                                                             down_scale, up_scale, mask_shape)
    radius = radius.astype(int)

    return femoral_head_mask_sequences, pivot, ch, cw, radius


def writeDicom(bones, ct, output, idx):
    # https://simpleitk.readthedocs.io/en/master/link_DicomSeriesReadModifyWrite_docs.html
    # https://simpleitk.readthedocs.io/en/master/link_DicomImagePrintTags_docs.html#lbl-print-image-meta-data-dictionary
    # https://simpleitk.readthedocs.io/en/v1.1.0/Examples/DicomSeriesReadModifyWrite/Documentation.html
    # https://blog.csdn.net/weixin_45069929/article/details/108690566
    # IMPORTANT: The CT must be series to get accurate meta data !!!

    # Transform from numpy to DICOM
    bones = bones[np.newaxis, :, :]
    filtered_image = sitk.GetImageFromArray(bones)
    filtered_image.SetSpacing(ct.GetSpacing())
    filtered_image.SetDirection(ct.GetDirection())

    # Initialize writter
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    modification_time = time.strftime('%H%M%S')
    modification_date = time.strftime('%Y%m%d')
    for k in ct.GetMetaDataKeys():
        if k != '0028|1052' and k != '0028|1053':
            filtered_image.SetMetaData(k, ct.GetMetaData(k))
    filtered_image.SetMetaData('0008|0031', modification_time)
    filtered_image.SetMetaData('0008|0021', modification_date)
    filtered_image.SetMetaData('0008|0008', 'DRIVED\\SECONDARY')

    # Set a unique UID to this image
    filtered_image.SetMetaData('0020|000e',
                               '1.2.840.113619.2.404.3.3233826508' + modification_date + '.1' + modification_time)
    writer.SetFileName(os.path.join(output, 'im{}'.format(idx) + '.dcm'))
    writer.Execute(filtered_image)


def seg_perform(pre_segs, pivot, femoral_head_masks, ch, cw, radius, process_type='left'):
    def find_bottom(width, mask):
        h_pos = np.where(mask[:, width] == 1)
        bottom_h = np.max(h_pos)
        return bottom_h

    h, w = pre_segs[0].shape
    cr = len(radius) // 2
    if process_type == 'left':
        pre_segs = [pre_segs[i][:, 0:w // 2] for i in range(len(pre_segs))]
    else:
        pre_segs = [np.flip(pre_segs[i][:, w // 2:], axis=1) for i in range(len(pre_segs))]

    sampled_pre_segs = pre_segs[pivot - cr:pivot + cr + 1]

    assert len(sampled_pre_segs) == len(femoral_head_masks)

    femoral_segs, acetabulum_segs = [], []
    for i in range(len(femoral_head_masks)):
        sampled_pre_seg = sampled_pre_segs[i]
        femoral_head_mask = femoral_head_masks[i]
        r = radius[i]
        valid_width = cw - r
        for width in range(cw, cw - r - 1, -1):
            valid_height = find_bottom(width, femoral_head_mask)
            if np.sum(sampled_pre_seg[valid_height:valid_height + r, width]) <= int(r / 3):
                valid_width = width
                break
        femoral_mask = np.zeros((h, w // 2))
        femoral_mask[:, 0:valid_width] = 1
        femoral_mask[:, valid_width:] = femoral_head_mask[:, valid_width:]
        femoral = sampled_pre_seg * femoral_mask
        acetabulum = sampled_pre_seg * (1 - femoral_mask)
        femoral_segs.append(femoral)
        acetabulum_segs.append(acetabulum)
    return femoral_segs, acetabulum_segs


def visualize_masks(output, series, femoral_masks, acetabulum_masks, pivot, process_type='left'):
    h, w = series[0].shape
    femoral_out_dir = os.path.join(output, process_type, 'vis_masks', 'fem')
    acetabulum_out_dir = os.path.join(output, process_type, 'vis_masks', 'ace')
    if not os.path.exists(femoral_out_dir):
        os.makedirs(femoral_out_dir)
    if not os.path.exists(acetabulum_out_dir):
        os.makedirs(acetabulum_out_dir)
    if process_type == 'right':
        femoral_masks = [np.flip(femoral_masks[i], axis=1) for i in range(len(femoral_masks))]
        acetabulum_masks = [np.flip(acetabulum_masks[i], axis=1) for i in range(len(acetabulum_masks))]
        cropped_series = series[:, :, w // 2:]
    else:
        cropped_series = series[:, :, 0:w // 2]
    cr = len(femoral_masks) // 2
    counter = 0
    for i in range(pivot - cr, pivot + cr + 1):
        femoral = cropped_series[i] * femoral_masks[counter]
        acetabulum = cropped_series[i] * acetabulum_masks[counter]
        scaled_femoral = (np.maximum(femoral, 0) / femoral.max()) * 255.0
        scaled_acetabulum = (np.maximum(acetabulum, 0) / acetabulum.max()) * 255.0
        counter += 1
        cv2.imwrite(os.path.join(femoral_out_dir, '{:05d}.png'.format(i)), scaled_femoral)
        cv2.imwrite(os.path.join(acetabulum_out_dir, '{:05d}.png'.format(i)), scaled_acetabulum)
    assert counter == len(femoral_masks) and len(femoral_masks) == len(acetabulum_masks)


def save_masks(pivot, output, masks, process_type, bone_type):
    assert process_type in ['left', 'right'], f'Invalid process type: {process_type}'
    assert bone_type in ['fem', 'ace'], f'Invalid bone type: {bone_type}'
    out_dir = os.path.join(output, process_type, 'masks', bone_type)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if process_type == 'right':
        masks = [np.flip(masks[i], axis=1) for i in range(len(masks))]
    cr = len(masks) // 2
    for i in range(len(masks)):
        mask = masks[i]
        idx = pivot - cr + i
        cv2.imwrite(os.path.join(out_dir, '{:05d}.png'.format(idx)), mask * 255)


def main(args):
    ct_series, pre_seg, output = args.ct_series, args.pre_seg, args.output
    sigma, radius = args.sigma, args.radius
    use_column, use_label = args.use_column, args.use_label
    left_initial_plane_idx, right_initial_plane_idx = args.left_initial_plane_idx, args.right_initial_plane_idx
    shrink = args.shrink
    calc_pre_seg = args.calc_pre_seg
    dilate, circle_peaks, percent = args.dilate, args.circle_peaks, args.percent
    down_scale, up_scale = args.down_scale, args.up_scale
    if not os.path.exists(output):
        os.makedirs(output)

    # Step 1: Read and blur the CT sequences
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(ct_series)
    dicom_names = dicom_names[::-1]
    reader.SetFileNames(dicom_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    images = reader.Execute()
    space_x, space_y, space_z = images.GetSpacing()
    assert space_x == space_y, f'Invalid space x {space_x} and space y {space_y}'
    series = sitk.GetArrayFromImage(images)
    blurred_ct = gaussian_filter(series, sigma=sigma, radius=radius)  # blur OK
    n, h, w = series.shape

    # Step 2: Get the pre-segmentation results
    pre_segs = read_masks(pre_seg)

    # Step 3: Get the ROIs from the CTs
    left_rois, left_ups, left_bottoms, right_rois, right_ups, right_bottoms = rois_selection_preSegs(pre_segs.copy(),
                                                                                                     use_column,
                                                                                                     use_label)

    # Step 4: Get the coarse segmentation in the ROIs
    if calc_pre_seg:
        # Result: the seg mask of the femoral head is smaller
        left_segs, right_segs = calc_pre_seg_func(blurred_ct, left_ups, left_bottoms, right_ups, right_bottoms)
    else:
        # Result: the seg mask of the femoral head is larger
        # Consistent improvement
        left_segs, right_segs = proj_segs(pre_segs, left_ups, left_bottoms, right_ups, right_bottoms)

    mask_shape = (h, w // 2)

    # Step 5: Determine the initial planes of the left and the right sides
    if left_initial_plane_idx > 0:
        left_initial_plane = left_segs[left_initial_plane_idx]
        # Step 6: Find the coarse anatomic circle in the femoral head
        left_circle_rr, left_circle_cc, left_ch, left_cw, left_radius = find_initial_circle_func(left_initial_plane,
                                                                                             shrink=args.shrink,
                                                                                             vis=True)
        # Step 7: Refine the coarse anatomy circle
        initial_left_up = left_ups[left_initial_plane_idx]
        initial_left_bottom = left_bottoms[left_initial_plane_idx]
        left_initial_array = series[left_initial_plane_idx, initial_left_up:initial_left_bottom, 0:w // 2]
        left_initial_blurred_array = blurred_ct[left_initial_plane_idx, initial_left_up:initial_left_bottom, 0:w // 2]
        left_contour, left_refined_ch, left_refined_cw, left_refined_cr = contour_refine_func(
            left_initial_blurred_array,
            left_initial_array, left_ch,
            left_cw, left_radius, shrink,
            dilate, percent, vis=True,
            circle_peaks=circle_peaks)
        # Step 8: Propagate the refined contour to the related slices that contain femoral head
        left_femoral_heads, left_pivot, left_ch, left_cw, left_radius = contour_prop(left_contour, left_refined_ch,
                                                                                     left_refined_cw, left_refined_cr,
                                                                                     space_x, space_z, left_ups,
                                                                                     left_bottoms, series,
                                                                                     blurred_ct, left_initial_plane_idx,
                                                                                     down_scale, up_scale, mask_shape)
        # Step 9: Impose the contours to the ct series
        left_femoral_segs, left_acetabulum_segs = seg_perform(pre_segs, left_initial_plane_idx, left_femoral_heads,
                                                              left_ch,
                                                              left_cw, left_radius, process_type='left')
        # Visualize for debug
        visualize_masks(output, series, left_femoral_segs, left_acetabulum_segs, left_initial_plane_idx,
                        process_type='left')
        # Save masks
        save_masks(left_initial_plane_idx, output, left_femoral_segs, 'left', 'fem')
        save_masks(left_initial_plane_idx, output, left_acetabulum_segs, 'left', 'ace')

    if right_initial_plane_idx > 0:
        right_initial_plane = right_segs[right_initial_plane_idx]


        # Horizontal inverse the right side (all processing are based on left side)
        hor_inv_right_initial_plane = np.flip(right_initial_plane, axis=1)

        # Step 6: Find the coarse anatomic circle in the femoral head
        right_circle_rr, right_circle_cc, right_ch, right_cw, right_radius = find_initial_circle_func(
            hor_inv_right_initial_plane, shrink=args.shrink)

        # Step 7: Refine the coarse anatomy circle
        initial_right_up = right_ups[right_initial_plane_idx]
        initial_right_bottom = right_bottoms[right_initial_plane_idx]

        right_initial_array = series[right_initial_plane_idx, initial_right_up:initial_right_bottom, w // 2:]
        right_initial_blurred_array = blurred_ct[right_initial_plane_idx, initial_right_up:initial_right_bottom, w // 2:]
        hor_inv_right_initial_array = np.flip(right_initial_array, axis=1)
        hor_inv_right_initial_blurred_array = np.flip(right_initial_blurred_array, axis=1)

        # Pass debug
        right_contour, right_refined_ch, right_refined_cw, right_refined_cr = contour_refine_func(
            hor_inv_right_initial_blurred_array, hor_inv_right_initial_array, right_ch, right_cw, right_radius, shrink,
            dilate, circle_peaks)

        # Step 8: Propagate the refined contour to the related slices that contain femoral head
        mask_shape = (h, w // 2)
        right_femoral_heads, right_pivot, right_ch, right_cw, right_radius = contour_prop(right_contour, right_refined_ch,
                                                                                          right_refined_cw,
                                                                                          right_refined_cr, space_x,
                                                                                          space_z, right_ups,
                                                                                          right_bottoms, series, blurred_ct,
                                                                                          right_initial_plane_idx,
                                                                                          down_scale, up_scale, mask_shape,
                                                                                          pros_type='right')

        # Step 9: Impose the contours to the ct series
        right_femoral_segs, right_acetabulum_segs = seg_perform(pre_segs, right_initial_plane_idx, right_femoral_heads,
                                                                right_ch, right_cw, right_radius, process_type='right')
        # Visualize for debug
        visualize_masks(output, series, right_femoral_segs, right_acetabulum_segs, right_initial_plane_idx,
                        process_type='right')
        # Save masks
        save_masks(right_initial_plane_idx, output, right_femoral_segs, 'right', 'fem')
        save_masks(right_initial_plane_idx, output, right_acetabulum_segs, 'right', 'ace')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ct_series', type=str, default='../../../bones/data/third_dicom/DICOM/PA104/ST0/SE2/')
    parser.add_argument('--pre_seg', type=str, default='../debug_rets/pa104_st0_se2_preSegs')
    parser.add_argument('--output', type=str, default='../debug_rets/pa104_st0_se2_debug_getSpace')

    # Blur args
    parser.add_argument('--sigma', type=float, default=1.5)
    parser.add_argument('--radius', type=int, default=4)

    # Roi selection args
    parser.add_argument('--use_column', type=int, default=0)
    parser.add_argument('--use_label', type=int, default=0)

    # Get pre segmentation results manually?
    parser.add_argument('--calc_pre_seg', type=int, default=0)

    # Initial plane of the left and right sides
    parser.add_argument('--left_initial_plane_idx', type=int, default=-1)  # 84, 103 for this sequence
    parser.add_argument('--right_initial_plane_idx', type=int, default=-1)

    # shrink parameter to find the best atonomy circle in the femoral head
    parser.add_argument('--shrink', type=float, default=1.7)

    # dilate and circle_peaks for the contour refinement
    parser.add_argument('--dilate', type=float, default=1.5)
    parser.add_argument('--circle_peaks', type=int, default=5)
    parser.add_argument('--percent', type=int, default=92)

    # down scale and up scale for determing the search range of hough transform
    parser.add_argument('--down_scale', type=float, default=0.8)
    parser.add_argument('--up_scale', type=float, default=1.2)

    args = parser.parse_args()
    main(args)
