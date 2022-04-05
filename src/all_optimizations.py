import enum
import os
import re
import sys
import json
import yaml
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
    parser.add_argument('--start_frame', type=int, default=1, help='The frame at which the optimized reconstruction will start.')
    parser.add_argument('--end_frame', type=int, default=-1, help='The frame at which the optimized reconstruction will end. If it is -1, start_frame and end_frame are automatically set.')
    parser.add_argument('--config', type=str, default='/configs/optimization.yaml', help='The path of a yaml config.')
    parser.add_argument('--ignore_cam', type=int, action='append', required=False, help='The camera index/indices to ignore for the trajectory estimation')
    parser.add_argument('--plot', action='store_true', help='Show the plots.')
    args = parser.parse_args()

    DATA_DIR = os.path.normpath(args.data_dir)
    assert os.path.exists(DATA_DIR), f'Data directory not found: {DATA_DIR}'

    # Load the config
    with open(args.config) as f:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        config = yaml.load(f, Loader=yaml.FullLoader)
    markers = config['marker']
    file_prefix = config['prefix']
    dlc_thresh = config['dlc_thresh']
    lure = 'lure' in markers
    skeletons = misc.get_skeleton(config['skeleton'], markers)
    body_label_dir = config['dlc']['body']
    lure_label_dir = config['dlc']['lure']

    # Load video info
    res, fps, num_frames, _ = app.get_vid_info(DATA_DIR)    # path to the directory having original videos
    vid_params = {
        'vid_resolution': res,
        'vid_fps': fps,
        'total_frames': num_frames,
    }
    assert 0 < args.start_frame < num_frames, f'start_frame must be strictly between 0 and {num_frames}'
    assert 0 != args.end_frame <= num_frames, f'end_frame must be less than or equal to {num_frames}'
    assert 0 <= dlc_thresh <= 1, 'dlc_thresh must be from 0 to 1'

    # load scene data
    k_arr, d_arr, r_arr, t_arr, cam_res, n_cams, scene_fpath = utils.find_scene_file(DATA_DIR, verbose=False)
    assert res == cam_res
    # load DLC data
    def _get_file_paths(dirname, mode='h5'):
        # generate labelled videos with DLC measurement data
        data = os.path.join(DATA_DIR, dirname)
        assert os.path.exists(data), f'Label directory not found: {data}'
        # load label data
        if mode == 'h5':
            return sorted(glob(os.path.join(data, '*.h5')))
        elif mode == 'pkl':
            return sorted(glob(os.path.join(data, 'cam*-predictions.pickle')))
        else:
            pass
    base_label_fpaths = _get_file_paths('dlc', mode='h5')
    dlc_pw_label_fpaths = _get_file_paths('dlc_pw', mode='pkl')
    body_label_fpaths = _get_file_paths(body_label_dir, mode='h5')
    lure_label_fpaths = _get_file_paths(lure_label_dir, mode='h5')
    assert n_cams == len(body_label_fpaths), f'# of dlc .h5 files != # of cams in {n_cams}_cam_scene_sba.json'
    assert n_cams == len(lure_label_fpaths), f'# of dlc .h5 files != # of cams in {n_cams}_cam_scene_sba.json'
    m = re.findall(r'cam([0-9]+)', ' '.join(body_label_fpaths))
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
        body_label_fpaths = del_elements(body_label_fpaths, ignore_cam_idx)
        lure_label_fpaths = del_elements(lure_label_fpaths, ignore_cam_idx)
    # prepare variables
    n_cams = len(k_arr)
    camera_params = (k_arr, d_arr, r_arr, t_arr, cam_res, cam_names, n_cams)

    # load labelled/DLC measurement dataframe (pixels, likelihood)
    body_points_2d_df = utils.load_dlc_points_as_df(body_label_fpaths, verbose=False)
    filtered_body_points_2d_df = body_points_2d_df.query(f'likelihood > {dlc_thresh}')    # ignore points with low likelihood
    lure_points_2d_df = utils.load_dlc_points_as_df(lure_label_fpaths, verbose=False)
    filtered_lure_points_2d_df = lure_points_2d_df.query(f'likelihood > {dlc_thresh}')    # ignore points with low likelihood

    # getting parameters
    if args.end_frame == -1:
        # Automatically set start and end frame
        # defining the first and end frame as detecting all the markers on any of cameras simultaneously
        def frame_condition_with_key_markers(point_2d_df, i: int, key_markers: List[str], n_min_cam: int) -> bool:
            markers_condition = ' or '.join([f'marker=="{ref}"' for ref in key_markers])
            markers = point_2d_df.query(
                f'frame == {i} and ({markers_condition})'
            )['marker']

            values, counts = np.unique(markers, return_counts=True)
            if len(values) != len(key_markers):
                return False

            return min(counts) >= n_min_cam

        # frames for cheetah
        start_frames = []
        end_frames = []
        max_idx = int(filtered_body_points_2d_df['frame'].max() + 1)
        for marker in markers:
            for i in range(max_idx):    # start_frame
                if frame_condition_with_key_markers(filtered_body_points_2d_df, i, [marker], 2):
                    start_frames.append(i)
                    break
            for i in range(max_idx, 0, -1): # end_frame
                if frame_condition_with_key_markers(filtered_body_points_2d_df, i, [marker], 2):
                    end_frames.append(i)
                    break
        if len(start_frames)==0 or len(end_frames)==0:
            raise('Setting frames failed. Please define start and end frames manually.')
        else:
            body_start_frame = max(start_frames)
            body_end_frame = min(end_frames)

        # frames for lure
        if lure:
            start_frames = []
            end_frames = []
            max_idx = int(filtered_lure_points_2d_df['frame'].max() + 1)
            for i in range(max_idx):    # start_frame
                if frame_condition_with_key_markers(filtered_lure_points_2d_df, i, ['lure'], 2):
                    start_frames.append(i)
                    break
            for i in range(max_idx, 0, -1): # end_frame
                if frame_condition_with_key_markers(filtered_lure_points_2d_df, i, ['lure'], 2):
                    end_frames.append(i)
                    break
            if len(start_frames)==0 or len(end_frames)==0:
                raise('Setting frames failed. Please define start and end frames manually.')
            else:
                lure_start_frame = max(start_frames)
                lure_end_frame = min(end_frames)
        else:
            lure_start_frame = body_start_frame
            lure_end_frame = body_end_frame
    else:
        # User-defined frames
        body_start_frame = lure_start_frame = args.start_frame - 1  # zero based indexing
        body_end_frame = lure_end_frame = args.end_frame % num_frames + 1 if args.end_frame == -1 else args.end_frame
    start_frame = max([body_start_frame, lure_start_frame])
    end_frame = min([body_end_frame, lure_end_frame])
    assert len(k_arr) == body_points_2d_df['camera'].nunique()
    assert len(k_arr) == lure_points_2d_df['camera'].nunique()
    assert body_start_frame < body_end_frame
    assert lure_start_frame < lure_end_frame
    assert start_frame < end_frame

    # print('========== DLC ==========\n')
    # _ = core.dlc(DATA_DIR, dlc_dir, markers, skeletons, dlc_thresh, params=vid_params, video=True)
    # print('========== Triangulation ==========\n')
    # core.tri(DATA_DIR, points_2d_df, 0, num_frames - 1, dlc_thresh, camera_params, scene_fpath, params=vid_params)
    # print('========== SBA ==========\n')
    # core.sba(DATA_DIR, points_2d_df, mode, camera_params, start_frame, end_frame, dlc_thresh, scene_fpath, params=vid_params, plot=args.plot, directions=True)
    # print('========== EKF ==========\n')
    # core.ekf(DATA_DIR, points_2d_df, mode, camera_params, start_frame, end_frame, dlc_thresh, scene_fpath, params=vid_params)
    print('========== FTE ==========\n')
    OUT_DIR = os.path.join(DATA_DIR, f'{file_prefix}_fte')
    pkl_fpath = core.fte(
        OUT_DIR,
        body_points_2d_df, lure_points_2d_df,
        config['FTE'],
        camera_params,
        markers,
        skeletons,
        start_frame, end_frame,
        body_start_frame, body_end_frame,
        lure_start_frame, lure_end_frame,
        dlc_thresh,
        scene_fpath,
        base_label_fpaths=base_label_fpaths,
        dlc_pw_label_fpaths=dlc_pw_label_fpaths,
        params=vid_params,
        enable_ppms=config['FTE']['enable_ppms'],
        video=True,
        plot=args.plot
    )