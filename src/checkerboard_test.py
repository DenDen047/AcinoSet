import os
import sys
import json
import numpy as np
import scipy
from numpy.core.defchararray import count
import itertools
import sympy as sp
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

from lib import misc, utils, app, vid, plotting, points, metric
from lib.calib import project_points_fisheye, triangulate_points_fisheye
from lib.misc import get_markers

import cv2 as cv


sns.set_theme()     # apply the default theme
plt.style.use(os.path.join('/configs', 'mplstyle.yaml'))

# metrics
def save_error_dists(pix_errors, output_dir: str):
    # variables
    errors = []
    for k, df in pix_errors.items():
        # errors.append(df['pixel_residual'].tolist())
        errors.append(df['error_u'].tolist() + df['error_v'].tolist())
    distances = []
    for k, df in pix_errors.items():
        distances += df['camera_distance'].tolist()

    # plot the error histogram
    xlabel = 'error (pix)'
    ylabel = 'freq'

    def _histogram(data, title, xlabel, ylabel, fpath):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        # histgram
        ax.hist(data, bins=20, density=True)

        # fit
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 1000)
        # normal distribution
        mu, sigma = scipy.stats.norm.fit(data)
        best_fit_line = scipy.stats.norm.pdf(x, mu, sigma)
        ax.plot(x, best_fit_line, label=r'normal ($\mu=${:.3f}, $\sigma=${:.3f})'.format(mu, sigma))
        # lognormal distribution
        sigma, loc, scale = scipy.stats.lognorm.fit(data)
        best_fit_line = scipy.stats.lognorm.pdf(x, sigma, loc, scale)
        ax.plot(x, best_fit_line, label=r'lognormal ($\sigma=${:.3f}, loc$=${:.3f}, scale$=${:.3f})'.format(sigma, loc, scale))

        # key values
        c = sns.color_palette()
        # median
        med = np.median(data)
        ax.axvline(med, color=c[1], linestyle='--', label='median={:.3f}'.format(med))
        # sigma-hat
        mad = scipy.stats.median_abs_deviation(data, scale='normal')
        ax.axvline(mad, color=c[2], linestyle='--', label='MAD={:.3f}'.format(mad))
        # three sigma
        sigma3 = 3 * sigma
        ax.axvline(sigma3, color=c[3], linestyle='--', label=r'$3\sigma=${:.3f}'.format(sigma3))
        # describe
        scores = pd.Series(data).describe()
        textstr = '\n'.join(['{}: {:.2f}'.format(idx, val) for idx, val in scores.iteritems()])

        # plot settings
        ax.set_title(title + ' (N={})'.format(len(data)))
        ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.text(
            0.8, 0.5,
            textstr,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        fig.savefig(fpath)

    _histogram(
        list(itertools.chain.from_iterable(errors)),
        title='Overall pixel errors',
        xlabel=xlabel, ylabel=ylabel,
        fpath=os.path.join(output_dir, "overall_error_hist.pdf")
    )

    hist_data = []
    labels = []
    for e, (k, df) in zip(errors, pix_errors.items()):
        i = int(k)
        hist_data.append(e)
        labels.append('cam{} (N={})'.format(i+1, len(e)))

        cam_name = i + 1
        _histogram(
            e,
            title='Camera{} pixel errors'.format(cam_name, len(e)),
            xlabel=xlabel, ylabel=ylabel,
            fpath=os.path.join(output_dir, "cam{}_error_hist.pdf".format(cam_name))
        )

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(hist_data, bins=20, histtype='bar')
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
        ax.scatter(d, e, alpha=0.5)
        errors += e
        distances += d
    coef = np.corrcoef(errors, distances)
    ax.set_title('All camera errors (N={}, coef={:.3f})'.format(len(errors), coef[0,1]))
    ax.set_xlabel('Radial distance between estimated 3D point and camera')
    ax.set_ylabel('Error (pix)')
    ax.legend([f'cam{str(i+1)}' for i in range(len(pix_errors))])
    fig.savefig(os.path.join(output_dir, "distance_vs_error.pdf"))


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


def get_sample(DATA_DIR, target_cam):
    calib_dir = os.path.join(DATA_DIR, 'extrinsic_calib')
    scene_fpath = os.path.join(calib_dir, '6_cam_scene_sba.json')
    frame_dir = os.path.join(calib_dir, 'frames', str(target_cam))
    points_fpaths = sorted(glob(os.path.join(calib_dir, 'points', 'points[1-9].json')))
    cam_i = target_cam - 1

    print('Target camera:', target_cam)

    # load scene
    k_arr, d_arr, r_arr, t_arr, cam_res = utils.load_scene(scene_fpath, verbose=False)
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
        pts_2d[a], frames[a],
        pts_2d[b], frames[b]
    )
    assert len(fnames) > 0

    # get target frame
    pprint(fnames)
    idx = 2
    target_frame_name = fnames[idx]
    img_pts_1 = img_pts_1[idx]
    img_pts_2 = img_pts_2[idx]
    print('Target frame:', target_frame_name)

    # load target frame
    target_frame_fpath = os.path.join(frame_dir, target_frame_name)
    image = cv.imread(target_frame_fpath, cv.IMREAD_COLOR)
    assert image is not None
    pts_3d = triangulate_points_fisheye(
        img_pts_1, img_pts_2,
        k_arr[a], d_arr[a], r_arr[a], t_arr[a],
        k_arr[b], d_arr[b], r_arr[b], t_arr[b]
    )

    # 3d to 2d
    pts_2d = project_points_fisheye(pts_3d, k_arr[cam_i], d_arr[cam_i], r_arr[cam_i], t_arr[cam_i])

    # draw
    # result = drawdots(image, pts_2d)
    result = drawdots(image, img_pts_1.reshape((-1, 2)))

    # save
    cv.imwrite(os.path.join(calib_dir, f'cam{target_cam}_'+target_frame_name), result)


def get_hist(DATA_DIR):
    calib_dir = os.path.join(DATA_DIR, 'extrinsic_calib')
    scene_fpath = os.path.join(calib_dir, '6_cam_scene_sba.json')
    points_fpaths = sorted(glob(os.path.join(calib_dir, 'points', 'points[1-9].json')))

    # load scene
    k_arr, d_arr, r_arr, t_arr, cam_res = utils.load_scene(scene_fpath, verbose=False)
    n_cams = len(k_arr)
    camera_params = (k_arr, d_arr, r_arr, t_arr, cam_res, n_cams)
    assert n_cams == len(points_fpaths)

    # load checkerboard points
    nx, ny = 6, 9
    checkerboard_info = {f'{x}_{y}': [] for x, y in itertools.product(range(nx), range(ny))}
    pts_2d = []
    cameras = []
    frames = []
    markers = []
    xs = []
    ys = []
    for cam_i, fpath in enumerate(points_fpaths):
        img_pts, img_names, *_ = utils.load_points(fpath)
        pts_2d.append(img_pts)
        frame = [int(s[3:8]) for s in img_names] # img file name -> frame number

        n = img_pts.shape[0]
        assert n == len(frame)

        for x, y in itertools.product(range(nx), range(ny)):
            frames += frame
            cameras += [cam_i] * n
            markers += [f'{x}_{y}'] * n
            xs += list(img_pts[:, y, x, 0])
            ys += list(img_pts[:, y, x, 1])

    points_2d_df = pd.DataFrame(
        np.array([frames, cameras, markers, xs, ys]).T,
        columns=['frame', 'camera', 'marker', 'x','y']
    )
    points_2d_df = points_2d_df.astype({'frame': 'int32', 'camera': 'int32', 'marker': 'str', 'x': 'float64', 'y': 'float64'})

    # estimate 3d position of checkerboard
    points_3d_df = utils.get_pairwise_3d_points_from_df(
        points_2d_df,
        k_arr, d_arr.reshape((-1,4)), r_arr, t_arr,
        triangulate_points_fisheye
    )

    # calculate residual error
    markers = points_2d_df['marker'].unique()
    pix_errors = metric.residual_error(points_2d_df, points_3d_df, markers, camera_params)
    save_error_dists(pix_errors, calib_dir)


# ========= MAIN ========
if __name__ == '__main__':
    parser = ArgumentParser(description='All Optimizations')
    parser.add_argument('--data_dir', type=str, help='The file path to the flick/run to be optimized.')
    args = parser.parse_args()

    DATA_DIR = os.path.normpath(args.data_dir)
    assert os.path.exists(DATA_DIR), f'Data directory not found: {DATA_DIR}'

    # _ = get_sample(
    #     DATA_DIR,
    #     target_cam=2
    # )
    _ = get_hist(DATA_DIR)
