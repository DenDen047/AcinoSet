import os
import sys
import json
import numpy as np
from numpy.core.defchararray import count
import sympy as sp
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt
from typing import Dict, List
from glob import glob
from time import time
from pprint import pprint
from tqdm import tqdm
from argparse import ArgumentParser
from scipy.stats import linregress
from pyomo.opt import SolverFactory

from lib import misc, utils, app, vid
from lib.calib import project_points_fisheye, triangulate_points_fisheye

import cv2 as cv


plt.style.use(os.path.join('/configs', 'mplstyle.yaml'))


def drawdots(img, pts, radius=3):
    n_pts = len(pts)
    colorclass = plt.cm.ScalarMappable(cmap='jet')
    C = colorclass.to_rgba(np.linspace(0, 1, n_pts))
    colors = (C[:, :3] * 255).astype(np.uint8).tolist()

    for i, pt in enumerate(pts):
        color = colors[i]
        img = cv.circle(
            img, tuple(pt.astype(np.uint16)),
            radius=radius, color=color, thickness=-1
        )
    return img


def epilines(DATA_DIR, points_2d_df, target_frame, target_bodypart, target_cam, epipolar_cam):
    OUT_DIR = os.path.join(DATA_DIR, 'epilines')
    os.makedirs(OUT_DIR, exist_ok=True)
    cam_i = target_cam - 1
    ecam_i = epipolar_cam - 1

    # filter DLC 2d points
    points_2d_df = points_2d_df.query(f'marker == "{target_bodypart}"')
    valid_frames = np.unique(points_2d_df['frame'])
    assert target_frame in valid_frames, print(f'choose from these frames: {valid_frames}')
    points_2d_df = points_2d_df[points_2d_df['frame'] == target_frame]


    # get video file paths
    candidate_cams = np.unique(points_2d_df['camera'])
    assert cam_i in candidate_cams, print(f'Please choose the camera index from {candidate_cams}')
    video_fpaths = sorted(glob(os.path.join(os.path.dirname(OUT_DIR), 'cam[1-9].mp4')))

    # load the original images
    clip1 = vid.VideoProcessorCV(in_name=video_fpaths[cam_i])
    clip2 = vid.VideoProcessorCV(in_name=video_fpaths[ecam_i])
    for _ in range(target_frame):
        clip1.load_frame()
        clip2.load_frame()
    image1 = clip1.load_frame()
    image2 = clip2.load_frame()

    # draw the DLC points
    pt = points_2d_df[points_2d_df['camera'] == cam_i][['x', 'y']].to_numpy()
    image1 = drawdots(image1, pt, radius=5)
    pt = points_2d_df[points_2d_df['camera'] == ecam_i][['x', 'y']].to_numpy()
    image2 = drawdots(image2, pt, radius=5)

    # get 3d points
    epipoint_pos = utils.get_pairwise_3d_points_from_df(
        points_2d_df,
        k_arr, d_arr.reshape((-1,4)), r_arr, t_arr,
        triangulate_points_fisheye,
        verbose=False
    )[['x', 'y', 'z']].to_numpy().flatten()
    camera_positions = misc.global_positions(r_arr, t_arr)
    epicam_pos = camera_positions[ecam_i].flatten()

    n = 1000
    points = []
    for i in range(n+1):
        r = i / n
        p = r*epicam_pos + (1-r)*epipoint_pos
        points.append(p)

    points_3d = np.vstack(points)

    # project 3d points to 2d ones
    pts1 = project_points_fisheye(points_3d, k_arr[cam_i], d_arr[cam_i], r_arr[cam_i], t_arr[cam_i])
    pts2 = project_points_fisheye(points_3d, k_arr[ecam_i], d_arr[ecam_i], r_arr[ecam_i], t_arr[ecam_i])

    # draw points
    result1 = drawdots(image1, pts1, radius=1)
    result2 = drawdots(image2, pts2, radius=1)

    # save
    cv.imwrite(os.path.join(OUT_DIR, f'frame_{target_frame}_{target_cam}.jpg'), result1)
    cv.imwrite(os.path.join(OUT_DIR, f'frame_{target_frame}_{epipolar_cam}.jpg'), result2)


def dlc(DATA_DIR, OUT_DIR, dlc_thresh, params: Dict = {}) -> Dict:
    video_fpaths = sorted(glob(os.path.join(DATA_DIR, 'cam[1-9].mp4'))) # original vids should be in the parent dir

    # save parameters
    params['dlc_thresh'] = dlc_thresh
    with open(os.path.join(OUT_DIR, 'video_params.json'), 'w') as f:
        json.dump(params, f)

    app.create_labeled_videos(video_fpaths, out_dir=OUT_DIR, draw_skeleton=True, pcutoff=dlc_thresh, lure=False)

    return params


# ========= MAIN ========
if __name__ == '__main__':
    parser = ArgumentParser(description='All Optimizations')
    parser.add_argument('--data_dir', type=str, help='The file path to the flick/run to be optimized.')
    parser.add_argument('--dlc_thresh', type=float, default=0.8, help='The likelihood of the dlc points below which will be excluded from the optimization.')
    args = parser.parse_args()

    mode = 'head'

    DATA_DIR = os.path.normpath(args.data_dir)
    assert os.path.exists(DATA_DIR), f'Data directory not found: {DATA_DIR}'

    # load video info
    res, fps, num_frames, _ = app.get_vid_info(DATA_DIR)    # path to the directory having original videos
    vid_params = {
        'vid_resolution': res,
        'vid_fps': fps,
        'total_frames': num_frames,
    }

    # generate labelled videos with DLC measurement data
    DLC_DIR = os.path.join(DATA_DIR, 'dlc_head')
    assert os.path.exists(DLC_DIR), f'DLC directory not found: {DLC_DIR}'
    # print('========== DLC ==========\n')
    # _ = dlc(DATA_DIR, DLC_DIR, args.dlc_thresh, params=vid_params)

    # load scene data
    # K ... The intrinsic matrix
    # D ... lens distortion
    # R, T ... the extrinsic parameters which denote the coordinate system transformations from 3D world coordinates to 3D camera coordinates
    k_arr, d_arr, r_arr, t_arr, cam_res, n_cams, scene_fpath = utils.find_scene_file(DATA_DIR, verbose=False)
    print('scene file:', scene_fpath)
    assert res == cam_res
    camera_params = (k_arr, d_arr, r_arr, t_arr, cam_res, n_cams)
    # load DLC data
    dlc_points_fpaths = sorted(glob(os.path.join(DLC_DIR, '*.h5')))
    assert n_cams == len(dlc_points_fpaths), f'# of dlc .h5 files != # of cams in {n_cams}_cam_scene_sba.json'

    # load measurement dataframe (pixels, likelihood)
    points_2d_df = utils.load_dlc_points_as_df(dlc_points_fpaths, frame_shifts=[0,0,1,0,0,-2], verbose=False)
    filtered_points_2d_df = points_2d_df.query(f'likelihood > {args.dlc_thresh}')    # ignore points with low likelihood

    assert len(k_arr) == points_2d_df['camera'].nunique()

    # Triangulation
    epilines(
        DATA_DIR, filtered_points_2d_df,
        target_frame=137,
        target_bodypart='r_eye',
        target_cam=2,
        epipolar_cam=3
    )

