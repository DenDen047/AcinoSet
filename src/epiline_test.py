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
from lib.misc import get_markers

import cv2 as cv


plt.style.use(os.path.join('/configs', 'mplstyle.yaml'))


def tri(DATA_DIR, points_2d_df, start_frame, end_frame, scene_fpath, target_frame, target_bodypart, target_cam, epipolar_cam, params: Dict = {}):
    OUT_DIR = os.path.join(DATA_DIR, 'tri')
    os.makedirs(OUT_DIR, exist_ok=True)
    cam_i = target_cam - 1
    ecam_i = epipolar_cam - 1
    # save reconstruction parameters
    params['start_frame'] = start_frame
    params['end_frame'] = end_frame
    with open(os.path.join(OUT_DIR, 'reconstruction_params.json'), 'w') as f:
        json.dump(params, f)

    # === Get Target Frame ===
    points_2d_df = points_2d_df.query(f'marker == "{target_bodypart}"')
    valid_frames = np.unique(points_2d_df['frame'])
    assert target_frame in valid_frames, print(f'choose from these frames: {valid_frames}')
    points_2d_df = points_2d_df[points_2d_df['frame'] == target_frame]

    # === Rendering ===
    # set parameters
    candidate_cams = np.unique(points_2d_df['camera'])
    assert cam_i in candidate_cams, print(f'Please choose the camera index from {candidate_cams}')
    video_fpaths = sorted(glob(os.path.join(os.path.dirname(OUT_DIR), 'cam[1-9].mp4')))

    # === Fundamental matrix ===
    points_3d = np.random.rand(1000, 3)
    r = 3
    points_3d[:, 0] = points_3d[:, 0] * r - r/2 + 8.0
    points_3d[:, 1] = points_3d[:, 1] * r - r/2 + 5.5
    points_3d[:, 2] = points_3d[:, 2] * r - r/2 + 0.0
    pts1 = project_points_fisheye(points_3d, k_arr[cam_i], d_arr[cam_i], r_arr[cam_i], t_arr[cam_i])
    pts2 = project_points_fisheye(points_3d, k_arr[ecam_i], d_arr[ecam_i], r_arr[ecam_i], t_arr[ecam_i])
    F, _ = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)

    # === Get Target Frame ===
    # 3d to 2d
    pts1 = points_2d_df[points_2d_df['camera'] == cam_i][['x', 'y']].to_numpy()
    pts2 = points_2d_df[points_2d_df['camera'] == ecam_i][['x', 'y']].to_numpy()
    n_points = len(pts1)

    # get epilines
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F)
    lines2 = lines2.reshape(-1, 3)

    # load the original image
    clip1 = vid.VideoProcessorCV(in_name=video_fpaths[cam_i])
    clip2 = vid.VideoProcessorCV(in_name=video_fpaths[ecam_i])
    for _ in range(target_frame):
        clip1.load_frame()
        clip2.load_frame()
    image1 = clip1.load_frame()
    image2 = clip2.load_frame()

    colorclass = plt.cm.ScalarMappable(cmap='jet')
    C = colorclass.to_rgba(np.linspace(0, 1, n_points))
    colors = (C[:, :3] * 255).astype(np.uint8).tolist()

    def drawlines(img1, lines, pts1):
        r, c, _ = img1.shape
        for i, (r,pt1) in enumerate(zip(lines,pts1)):
            color = colors[i]
            x0,y0 = map(int, [0, -r[2]/r[1] ])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
            img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
            img1 = cv.circle(img1,tuple(pt1.astype(np.uint16)),5,color,-1)
        return img1

    result1= drawlines(image1, lines1, pts1)
    result2 = drawlines(image2 ,lines2, pts2)

    # save
    cv.imwrite(os.path.join(OUT_DIR, f'frame_{target_frame}_1.jpg'), result1)
    cv.imwrite(os.path.join(OUT_DIR, f'frame_{target_frame}_2.jpg'), result2)


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
    parser.add_argument('--start_frame', type=int, default=1, help='The frame at which the optimized reconstruction will start.')
    parser.add_argument('--end_frame', type=int, default=-1, help='The frame at which the optimized reconstruction will end. If it is -1, start_frame and end_frame are automatically set.')
    parser.add_argument('--dlc_thresh', type=float, default=0.8, help='The likelihood of the dlc points below which will be excluded from the optimization.')
    parser.add_argument('--plot', action='store_true', help='Show the plots.')
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
    assert 0 < args.start_frame < num_frames, f'start_frame must be strictly between 0 and {num_frames}'
    assert 0 != args.end_frame <= num_frames, f'end_frame must be less than or equal to {num_frames}'
    assert 0 <= args.dlc_thresh <= 1, 'dlc_thresh must be from 0 to 1'

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
    _ = tri(
        DATA_DIR, filtered_points_2d_df, 0, num_frames - 1, scene_fpath,
        target_frame=137,
        target_bodypart='r_eye',
        target_cam=2,
        epipolar_cam=3,
        params=vid_params
    )
    plt.close('all')

    if args.plot:
        print('Plotting results...')
        data_fpaths = [
            os.path.join(DATA_DIR, 'tri', 'tri.pickle'),    # plot is too busy when tri is included
            os.path.join(DATA_DIR, 'sba', 'sba.pickle'),
            os.path.join(DATA_DIR, 'ekf', 'ekf.pickle'),
            os.path.join(DATA_DIR, 'fte', 'fte.pickle')
        ]
        app.plot_multiple_cheetah_reconstructions(data_fpaths, reprojections=False, dark_mode=True)
