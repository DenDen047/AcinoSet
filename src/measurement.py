import os
import sys
import json
import pickle
import numpy as np
import sympy as sp
from scipy.spatial import distance
from scipy.optimize import least_squares
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

from lib import misc, utils, app, metric
from lib.calib import project_points_fisheye, triangulate_points_fisheye
from lib.misc import get_markers, rot_x, rot_y, rot_z

import core


sns.set_theme()     # apply the default theme
plt.style.use(os.path.join('/configs', 'mplstyle.yaml'))


def save_error_dists(pix_errors, output_dir: str) -> float:
    # variables
    errors = []
    for k, df in pix_errors.items():
        errors += df['pixel_residual'].tolist()
    distances = []
    for k, df in pix_errors.items():
        distances += df['camera_distance'].tolist()

    # plot the error histogram
    xlabel = 'error (pix)'
    ylabel = 'freq'

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(errors)
    ax.set_title('Overall pixel errors (N={})'.format(len(errors)))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.savefig(os.path.join(output_dir, "overall_error_hist.pdf"))

    hist_data = []
    labels = []
    for k, df in pix_errors.items():
        i = int(k)
        e = df['pixel_residual'].tolist()
        hist_data.append(e)
        labels.append('cam{} (N={})'.format(i+1, len(e)))

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(hist_data, bins=10, density=True, histtype='bar')
    ax.legend(labels)
    ax.set_title('Reprojection pixel errors')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.savefig(os.path.join(output_dir, "cams_error_hist.pdf"))

    # the relation between camera distance and pixel errors
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    errors = []
    distances = []
    for k, df in pix_errors.items():
        e = df['pixel_residual'].tolist()
        d = df['camera_distance'].tolist()
        ax.scatter(e, d, alpha=0.5)
        errors += e
        distances += d
    coef = np.corrcoef(errors, distances)
    ax.set_title('All camera errors (N={}, coef={:.3f})'.format(len(errors), coef[0,1]))
    ax.set_xlabel('Radial distance between estimated 3D point and camera')
    ax.set_ylabel('Error (pix)')
    ax.legend([f'cam{str(i+1)}' for i in range(len(pix_errors))])
    fig.savefig(os.path.join(output_dir, "distance_vs_error.pdf"))

    return np.mean(errors)


def tri(DATA_DIR, points_2d_df, start_frame, end_frame, dlc_thresh, camera_params, scene_fpath, params: Dict = {}) -> str:
    OUT_DIR = os.path.join(DATA_DIR, 'tri')
    os.makedirs(OUT_DIR, exist_ok=True)
    markers = misc.get_markers(mode='all')
    k_arr, d_arr, r_arr, t_arr, _, _ = camera_params

    # get the camera positions
    # for c in range(len(r_arr)):
    #     R = r_arr[c].T
    #     t = t_arr[c]
    #     T = -R @ t
    #     print(T)

    # save reconstruction parameters
    params['start_frame'] = start_frame
    params['end_frame'] = end_frame
    params['dlc_thresh'] = dlc_thresh
    with open(os.path.join(OUT_DIR, 'reconstruction_params.json'), 'w') as f:
        json.dump(params, f)

    # triangulation
    points_2d_df = points_2d_df.query(f'likelihood > {dlc_thresh}')
    points_2d_df = points_2d_df[points_2d_df['frame'].between(start_frame, end_frame)]
    points_3d_df = utils.get_pairwise_3d_points_from_df(
        points_2d_df,
        k_arr, d_arr.reshape((-1,4)), r_arr, t_arr,
        triangulate_points_fisheye
    )
    points_3d_df['point_index'] = points_3d_df.index

    # calculate residual error
    pix_errors = metric.residual_error(points_2d_df, points_3d_df, markers, camera_params)
    save_error_dists(pix_errors, OUT_DIR)

    # measure the specific parameters
    n_frame = max(points_3d_df['frame'].unique()) + 1
    data = {
        'head': [],
        'spine': [],
    }
    for f in range(n_frame):
        points_df = points_3d_df.query(f'frame == {f}')

        # head position
        head = []
        head_df = points_df.query('marker == "nose" | marker == "r_eye" | marker == "l_eye"')
        if len(head_df['marker'].unique()) == 3:
            nose = head_df.query('marker == "nose"')[['x', 'y', 'z']].to_numpy().flatten()
            l_eye = head_df.query('marker == "l_eye"')[['x', 'y', 'z']].to_numpy().flatten()
            r_eye = head_df.query('marker == "r_eye"')[['x', 'y', 'z']].to_numpy().flatten()
            coe = np.mean([l_eye, r_eye], axis=0)

            # get the head position and pose with least square method
            def func(x):
                p_head = np.array([x[0], x[1], x[2]]) # [x,y,z]
                phi, psi, theta = x[3], x[4], x[5]

                RI_0  = rot_z(psi) @ rot_x(phi) @ rot_y(theta)  # head
                R0_I  = RI_0.T

                p_l_eye = p_head + R0_I @ np.array([0, 0.038852231676497324, 0])
                p_r_eye = p_head + R0_I @ np.array([0, -0.038852231676497324, 0])
                p_nose  = p_head + R0_I @ np.array([0.0571868749393016, 0, -0.0571868749393016])
                return np.array([
                    np.linalg.norm(l_eye - p_l_eye),
                    np.linalg.norm(r_eye - p_r_eye),
                    np.linalg.norm(nose - p_nose),
                ])
            r = least_squares(
                fun=func,
                x0=np.array([coe[0], coe[1], coe[2], 0, 0, 0])
            )
            head_pos = r.x[:3]
        else:
            head_pos = [np.nan] * 3
        data['head'].append(head_pos)

        # spine position
        spine_df = points_df.query('marker == "spine"')
        data['spine'].append(
            spine_df[['x', 'y', 'z']].to_numpy().flatten()
            if len(spine_df['marker'].unique()) == 1
            else [np.nan] * 3
        )

    for k, v in data.items():
        data[k] = np.array(v)

    fig_fpath = os.path.join(OUT_DIR, 'summary.pdf')
    app.plot_key_positions(data, out_fpath=fig_fpath)

    # ========= SAVE TRIANGULATION RESULTS ========
    positions = np.full((end_frame - start_frame + 1, len(markers), 3), np.nan)

    for i, marker in enumerate(markers):
        marker_pts = points_3d_df[points_3d_df['marker']==marker][['frame', 'x', 'y', 'z']].values
        for frame, *pt_3d in marker_pts:
            positions[int(frame) - start_frame, i] = pt_3d

    out_fpath = app.save_tri(positions, OUT_DIR, scene_fpath, markers, start_frame, pix_errors, save_videos=False)

    return out_fpath


# ========= MAIN ========
if __name__ == '__main__':
    parser = ArgumentParser(description='All Optimizations')
    parser.add_argument('--data_dir', type=str, help='The file path to the flick/run to be optimized.')
    parser.add_argument('--start_frame', type=int, default=1, help='The frame at which the optimized reconstruction will start.')
    parser.add_argument('--end_frame', type=int, default=-1, help='The frame at which the optimized reconstruction will end. If it is -1, start_frame and end_frame are automatically set.')
    parser.add_argument('--dlc_thresh', type=float, default=0.8, help='The likelihood of the dlc points below which will be excluded from the optimization.')
    parser.add_argument('--plot', action='store_true', help='Show the plots.')
    args = parser.parse_args()

    mode = 'head_stabilize'

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
    DLC_DIR = os.path.join(DATA_DIR, 'dlc')
    assert os.path.exists(DLC_DIR), f'DLC directory not found: {DLC_DIR}'
    # print('========== DLC ==========\n')
    # _ = core.dlc(DATA_DIR, DLC_DIR, args.dlc_thresh, params=vid_params)

    # load scene data
    k_arr, d_arr, r_arr, t_arr, cam_res, n_cams, scene_fpath = utils.find_scene_file(DATA_DIR, verbose=False)
    assert res == cam_res
    camera_params = (k_arr, d_arr, r_arr, t_arr, cam_res, n_cams)
    # load DLC data
    dlc_points_fpaths = sorted(glob(os.path.join(DLC_DIR, '*.h5')))
    assert n_cams == len(dlc_points_fpaths), f'# of dlc .h5 files != # of cams in {n_cams}_cam_scene_sba.json'

    # load measurement dataframe (pixels, likelihood)
    points_2d_df = utils.load_dlc_points_as_df(dlc_points_fpaths, frame_shifts=[0,0,0,0,0,0], verbose=False)
    filtered_points_2d_df = points_2d_df.query(f'likelihood > {args.dlc_thresh}')    # ignore points with low likelihood

    # getting parameters
    if args.end_frame == -1:
        # Automatically set start and end frame
        # defining the first and end frame as detecting all the markers on any of cameras simultaneously
        target_markers = misc.get_markers(mode)

        def frame_condition(i: int, target_markers: List[str]) -> bool:
            markers_condition = ' or '.join([f'marker=="{ref}"' for ref in target_markers])
            num_marker = lambda i: len(filtered_points_2d_df.query(f'frame == {i} and ({markers_condition})')['marker'].unique())
            return num_marker(i) >= len(target_markers)

        def frame_condition_with_key_markers(i: int, key_markers: List[str], n_min_cam: int) -> bool:
            markers_condition = ' or '.join([f'marker=="{ref}"' for ref in key_markers])
            markers = filtered_points_2d_df.query(
                f'frame == {i} and ({markers_condition})'
            )['marker']

            values, counts = np.unique(markers, return_counts=True)
            if len(values) != len(key_markers):
                return False

            return min(counts) >= n_min_cam

        start_frame, end_frame = None, None
        max_idx = int(filtered_points_2d_df['frame'].max() + 1)
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

    print('start frame:', start_frame)
    print('end frame:', end_frame)

    pkl_fpath = os.path.join(DATA_DIR, 'fte', 'fte.pickle')
    if not os.path.exists(pkl_fpath):
        print('========== FTE ==========\n')
        pkl_fpath = core.fte(DATA_DIR, points_2d_df, mode, camera_params, start_frame, end_frame, args.dlc_thresh, scene_fpath, params=vid_params, shutter_delay=True, interpolation_mode='acc', plot=args.plot)

    # load pickle data
    with open(pkl_fpath, 'rb') as f:
        data = pickle.load(f)
    positions_3d = np.array(data['positions'])    # [n_cam, n_frame, n_label, xyz]
    positions_3d = positions_3d[0]  # cam1 is the base

    # measure the specific parameters
    labels = misc.get_markers(mode)
    n_frame = positions_3d.shape[0]
    data = {
        'head z': [],
        'spine z': [],
        'neck_base z': [],
        'neck_length': [],
    }
    for f in range(n_frame):
        labels_position = positions_3d[f, :, :]

        # head position
        l_eye = labels_position[labels.index('l_eye'), :]
        r_eye = labels_position[labels.index('r_eye'), :]
        head = np.mean([l_eye, r_eye], axis=0)
        data['head z'].append(head[2])

        # spine position
        spine = labels_position[labels.index('spine'), :]
        data['spine z'].append(spine[2])

        # neck_base position
        neck_base = labels_position[labels.index('neck_base'), :]
        data['neck_base z'].append(neck_base[2])

        # neck length
        neck_length = np.linalg.norm(head - neck_base)
        data['neck_length'].append(neck_length)

    for k, v in data.items():
        data[k] = np.array(v)

    fig_fpath = os.path.join(DATA_DIR, 'fte', 'summary.pdf')
    app.plot_key_values(data, out_fpath=fig_fpath)
