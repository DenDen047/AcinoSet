import enum
import os
import re
import sys
import json
import numpy as np
import sympy as sp
from scipy.spatial import distance
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from glob import glob
from time import time
from pprint import pprint
from tqdm import tqdm
from argparse import ArgumentParser
from scipy.stats import linregress
from pyomo.opt import SolverFactory
from skspatial.objects import Plane, Points
from skspatial.plotting import plot_3d

from lib import misc, utils, app, metric
from lib.calib import project_points_fisheye, triangulate_points_fisheye
from lib.misc import get_markers

import core


sns.set_theme()     # apply the default theme
plt.style.use(os.path.join('/configs', 'mechatronics_style.yaml'))


# ========= MAIN ========
if __name__ == '__main__':
    parser = ArgumentParser(description='All Optimizations')
    parser.add_argument('--data_dir', type=str, help='The file path to the flick/run to be optimized.')
    parser.add_argument('--dlc', type=str, default='dlc', help='The file path to the flick/run to be optimized.')
    parser.add_argument('--start_frame', type=int, default=1, help='The frame at which the optimized reconstruction will start.')
    parser.add_argument('--end_frame', type=int, default=-1, help='The frame at which the optimized reconstruction will end. If it is -1, start_frame and end_frame are automatically set.')
    parser.add_argument('--dlc_thresh', type=float, default=0.8, help='The likelihood of the dlc points below which will be excluded from the optimization.')
    parser.add_argument('--ignore_cam', type=int, action='append', required=False, help='The camera index/indices to ignore for the trajectory estimation')
    args = parser.parse_args()

    mode = 'upper_body'

    DATA_DIR = os.path.normpath(args.data_dir)
    assert os.path.exists(DATA_DIR), f'Data directory not found: {DATA_DIR}'

    # load video info
    res, fps, num_frames, _ = app.get_vid_info(DATA_DIR)    # path to the directory having original videos
    vid_params = {
        'vid_resolution': res,
        'vid_fps': fps,
        'total_frames': num_frames,
    }
    assert 0 < args.start_frame < num_frames, f'start_frame must be strictly between 0 and {num_frames}'
    assert 0 != args.end_frame <= num_frames, f'end_frame must be less than or equal to {num_frames}'
    assert 0 <= args.dlc_thresh <= 1, 'dlc_thresh must be from 0 to 1'

    # generate labelled videos with DLC measurement data
    DLC_DIR = os.path.join(DATA_DIR, args.dlc)
    assert os.path.exists(DLC_DIR), f'DLC directory not found: {DLC_DIR}'

    # load scene data
    k_arr, d_arr, r_arr, t_arr, cam_res, n_cams, scene_fpath = utils.find_scene_file(DATA_DIR, verbose=False)
    assert res == cam_res
    # load DLC data
    dlc_points_fpaths = sorted(glob(os.path.join(DLC_DIR, '*.h5')))
    assert n_cams == len(dlc_points_fpaths), f'# of dlc .h5 files != # of cams in {n_cams}_cam_scene_sba.json'
    m = re.findall(r'cam([0-9]+)', ' '.join(dlc_points_fpaths))
    cam_names = [i for i in m] # 1-6
    # ignored cameras
    if args.ignore_cam is not None and len(args.ignore_cam) > 0:
        def del_elements(arr, indices):
            if type(arr) is list:
                return [a for i, a in enumerate(arr) if i not in indices]
            elif type(arr) is np.ndarray:
                indices = [i for i in range(len(arr)) if i not in indices]
                return arr[indices, :]
        ignore_cam_idx = [i for i, c in enumerate(cam_names) if int(c) in args.ignore_cam] # 0-5
        k_arr = del_elements(k_arr, ignore_cam_idx)
        d_arr = del_elements(d_arr, ignore_cam_idx)
        r_arr = del_elements(r_arr, ignore_cam_idx)
        t_arr = del_elements(t_arr, ignore_cam_idx)
        cam_names = del_elements(cam_names, ignore_cam_idx)
        dlc_points_fpaths = del_elements(dlc_points_fpaths, ignore_cam_idx)
    # prepare variables
    n_cams = len(k_arr)
    camera_params = (k_arr, d_arr, r_arr, t_arr, cam_res, cam_names, n_cams)

    # load DLC measurement dataframe (pixels, likelihood)
    points_2d_df = utils.load_dlc_points_as_df(dlc_points_fpaths, verbose=False)
    # filtered_points_2d_df = points_2d_df.query(f'likelihood > {args.dlc_thresh}')    # ignore points with low likelihood

    # getting parameters
    if args.end_frame == -1:
        # Automatically set start and end frame
        # defining the first and end frame as detecting all the markers on any of cameras simultaneously
        target_markers = misc.get_markers(mode)

        def frame_condition(i: int, target_markers: List[str]) -> bool:
            markers_condition = ' or '.join([f'marker=="{ref}"' for ref in target_markers])
            num_marker = lambda i: len(points_2d_df.query(f'frame == {i} and ({markers_condition})')['marker'].unique())
            return num_marker(i) >= len(target_markers)

        def frame_condition_with_key_markers(i: int, key_markers: List[str], n_min_cam: int) -> bool:
            markers_condition = ' or '.join([f'marker=="{ref}"' for ref in key_markers])
            markers = points_2d_df.query(
                f'frame == {i} and ({markers_condition})'
            )['marker']

            values, counts = np.unique(markers, return_counts=True)
            if len(values) != len(key_markers):
                return False

            return min(counts) >= n_min_cam

        start_frame, end_frame = None, None
        max_idx = int(points_2d_df['frame'].max() + 1)
        for i in range(max_idx):    # start_frame
            if frame_condition_with_key_markers(i, target_markers, 2):
            # if frame_condition(i, target_markers):
                start_frame = i
                break
        for i in range(max_idx, 0, -1): # end_frame
            if frame_condition_with_key_markers(i, target_markers, 2):
            # if frame_condition(i, target_markers):
                end_frame = i
                break
        if start_frame is None or end_frame is None:
            raise('Setting frames failed. Please define start and end frames manually.')
    else:
        # User-defined frames
        start_frame = args.start_frame - 1  # 0 based indexing
        end_frame = args.end_frame % num_frames + 1 if args.end_frame == -1 else args.end_frame
    assert len(k_arr) == points_2d_df['camera'].nunique()
    assert start_frame != end_frame

    # plot the transition of likelihoods
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("tab10")
    cam = 1
    points_2d_df = points_2d_df.query('camera == {}'.format(cam - 1))

    markers = misc.get_markers(mode)
    for i, marker in enumerate(markers):
        df = points_2d_df.query('marker == "{}"'.format(marker))
        print(df['frame'].min(), df['frame'].max())
        df.plot(
            x='frame', y='likelihood',
            # s=3,
            c=cmap(i),
            ax=ax, label=marker
        )

    # show other info
    ax.axhline(y=args.dlc_thresh, linestyle='--')
    ax.axvline(x=start_frame)
    ax.axvline(x=end_frame)
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Likelihood')
    ax.set_title('{} (camera {})'.format(DLC_DIR, cam))
    ax.legend()

    # save
    output_fpath = os.path.join(DLC_DIR, 'frame_likelihood.pdf')
    fig.savefig(output_fpath)
