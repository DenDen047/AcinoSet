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

from lib import misc, utils, app, vid, plotting, points
from lib.calib import project_points_fisheye, triangulate_points_fisheye
from lib.misc import get_markers

import cv2 as cv


plt.style.use(os.path.join('/configs', 'mplstyle.yaml'))


def checkerboard(DATA_DIR, target_frame, target_bodypart, target_cam):
    calib_dir = os.path.join(DATA_DIR, 'extrinsic_calib')
    scene_fpath = os.path.join(calib_dir, '6_cam_scene_sba.json')
    frame_dir = os.path.join(calib_dir, 'frames', str(target_cam))
    points_fpaths = sorted(glob(os.path.join(calib_dir, 'points', 'points[1-9].json')))
    target_frame_name = 'img{:05}.jpg'.format(target_frame)
    target_frame_fpath = os.path.join(frame_dir, target_frame_name)
    cam_i = target_cam - 1

    # load scene
    k_arr, d_arr, r_arr, t_arr, cam_res = utils.load_scene(scene_fpath)
    n_cams = len(k_arr)

    pts_2d = []
    frames = []
    for fpath in points_fpaths:
        img_pts, img_names, *_ = utils.load_points(fpath)
        pts_2d.append(img_pts)
        frames.append(img_names)

    # estimate 3d position of checkerboard
    a = cam_i
    b = (cam_i + 1) % n_cams
    img_pts_1, img_pts_2, fnames = points.common_image_points(
        pts_2d[a], [target_frame_name],
        pts_2d[b], frames[b]
    )
    assert len(fnames) > 0
    pts_3d = triangulate_points_fisheye(
        img_pts_1, img_pts_2,
        k_arr[a], d_arr[a], r_arr[a], t_arr[a],
        k_arr[b], d_arr[b], r_arr[b], t_arr[b]
    )

    # 3d to 2d
    pts_2d = project_points_fisheye(pts_3d, k_arr[cam_i], d_arr[cam_i], r_arr[cam_i], t_arr[cam_i])
    print(pts_2d.shape)

    # load the frame
    image = cv.imread(target_frame_fpath, cv.IMREAD_COLOR)
    assert image is not None

    # draw
    def drawdots(img, pts):
        n_pts = len(pts)
        colorclass = plt.cm.ScalarMappable(cmap='jet')
        C = colorclass.to_rgba(np.linspace(0, 1, n_pts))
        colors = (C[:, :3] * 255).astype(np.uint8).tolist()

        for i, pt1 in enumerate(pts):
            color = colors[i]
            img = cv.circle(img, tuple(pt1.astype(np.uint16)), 5, color, -1)
        return img

    result = drawdots(image, pts_2d)

    # save
    cv.imwrite(os.path.join(calib_dir, f'cam{target_cam}_{target_frame}.jpg'), result)


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

    DATA_DIR = os.path.normpath(args.data_dir)
    assert os.path.exists(DATA_DIR), f'Data directory not found: {DATA_DIR}'

    # Checkerboard
    _ = checkerboard(
        DATA_DIR,
        target_frame=197,
        target_bodypart='r_eye',
        target_cam=2,
        frame_shifts=[0,0,1,0,0,-2]
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
