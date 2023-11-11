import os
import argparse
from icecream import ic
import pickle
from shutil import rmtree


def read_plane_indices_file(idx_file):
    ret = {}
    with open(idx_file, 'r') as f:
        indices = f.readlines()
        for i in range(len(indices)):
            items = indices[i].split(': ')
            ret[items[0]] = int(items[1][:-1])
    return ret


def main(args):
    counter = 0
    patients = sorted(os.listdir(args.root), key=lambda x: int(x[2:]))
    left_plane_indices = read_plane_indices_file(args.left_initial_plane_idx_file)
    right_plane_indices = read_plane_indices_file(args.right_initial_plane_idx_file)
    max_ct_sequences = max(len(left_plane_indices), len(right_plane_indices))
    if args.group is not None:
        f = open(args.group, 'rb')
        disorder_sequences = pickle.load(f)
        f.close()
    for p in patients:
        p_dir = os.path.join(args.root, p)
        sections = sorted(os.listdir(p_dir), key=lambda x: int(x[2:]))
        for s in sections:
            se_dir = os.path.join(p_dir, s)
            ses = sorted(os.listdir(se_dir), key=lambda x: int(x[2:]))
            for e in ses:
                if args.group is not None:
                    if f'{p}_{s}_{e}' not in disorder_sequences:
                        continue
                img_dir = os.path.join(se_dir, e)
                output_dir = os.path.join(args.out_dir, p, s, e)
                preSeg_dir = os.path.join(args.pre_seg, p, s, e)
                if os.path.exists(output_dir):
                    rmtree(output_dir)
                if f'{p}_{s}_{e}' in left_plane_indices:
                    left_initial_plane_idx = left_plane_indices[f'{p}_{s}_{e}'] - 1
                else:
                    left_initial_plane_idx = -1
                if f'{p}_{s}_{e}' in right_plane_indices:
                    right_initial_plane_idx = right_plane_indices[f'{p}_{s}_{e}'] - 1
                else:
                    right_initial_plane_idx = -1
                if left_initial_plane_idx == -1 and right_initial_plane_idx == -1:
                    continue
                command = 'python instance_seg_main.py --ct_series {} --pre_seg {} --output {} --sigma {} --radius {} --use_column {} --use_label {} ' \
                          '--calc_pre_seg {} --left_initial_plane_idx {} --right_initial_plane_idx {} --shrink {} ' \
                          '--dilate {} --circle_peaks {} --percent {} --down_scale {} --up_scale {}'.format(
                    img_dir, preSeg_dir, output_dir, args.sigma, args.radius, args.use_column, args.use_label,
                    args.calc_pre_seg, left_initial_plane_idx, right_initial_plane_idx, args.shrink,
                    args.dilate, args.circle_peaks, args.percent, args.down_scale, args.up_scale)
                counter += 1
                print('[{}]/[{}] CT sequence is processed'.format(counter, max_ct_sequences))
                ic(command)
                os.system(command)
                # temp = input('kkpsa')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../../../bones/data/dicom/')
    parser.add_argument('--pre_seg', type=str, default='../pre_seg_masks/serie1')
    parser.add_argument('--out_dir', type=str, default='../instance_masks/serie1')

    # Blur args
    parser.add_argument('--sigma', type=float, default=1.5)
    parser.add_argument('--radius', type=int, default=4)

    # Roi selection args
    parser.add_argument('--use_column', type=int, default=0)
    parser.add_argument('--use_label', type=int, default=0)

    # Get pre segmentation results manually?
    parser.add_argument('--calc_pre_seg', type=int, default=0)

    # Initial plane of the left and right sides
    parser.add_argument('--left_initial_plane_idx_file', type=str, default='serie1_left.txt')
    parser.add_argument('--right_initial_plane_idx_file', type=str, default='serie1_right.txt')

    # shrink parameter to find the best atonomy circle in the femoral head
    parser.add_argument('--shrink', type=float, default=1.7)

    # dilate and circle_peaks for the contour refinement
    parser.add_argument('--dilate', type=float, default=1.5)
    parser.add_argument('--circle_peaks', type=int, default=5)
    parser.add_argument('--percent', type=int, default=92)

    # down scale and up scale for determing the search range of hough transform
    parser.add_argument('--down_scale', type=float, default=0.8)
    parser.add_argument('--up_scale', type=float, default=1.2)

    parser.add_argument('--group', type=str, default=None)

    args = parser.parse_args()
    main(args)
