#!/usr/bin/env bash

# serie 1
python main_warpper.py

# serie 2
python main_warpper.py --root ../../../bones/data/second_dicom --pre_seg ../pre_seg_masks/serie2 --out_dir ../instance_masks/serie2 --left_initial_plane_idx_file serie2_left.txt --right_initial_plane_idx_file serie2_right.txt

# serie 3
python main_warpper.py --root ../../../bones/data/third_dicom/DICOM --pre_seg ../pre_seg_masks/serie3 --out_dir ../instance_masks/serie3 --left_initial_plane_idx_file serie3_left.txt --right_initial_plane_idx_file serie3_right.txt