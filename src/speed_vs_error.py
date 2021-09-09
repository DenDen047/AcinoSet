import os
import sys
import json
import pickle
import numpy as np
import sympy as sp
from scipy import stats
from scipy.spatial import distance
from scipy.optimize import least_squares
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt
from matplotlib import rc
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
plt.style.use(os.path.join('/configs', 'mechatronics_style.yaml'))

rc('text', usetex=True)


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
    sd = False
    sd_mode = 'const'
    intermode = 'pos'

    data_fpahts = [
        '/data/2017_08_29/top/phantom/run1_1/',
        '/data/2017_12_21/top/lily/run1/',
        '/data/2019_03_05/lily/run/',
        '/data/2019_03_09/lily/run/',
    ]

    pkl_fpaths = {
        'pos': [],
        'vel': [],
        'acc': [],
    }
    for data_fpath in data_fpahts:
        pkl_fpaths['pos'].append(os.path.join(data_fpath, 'fte_baseline', 'fte.pickle'))
        pkl_fpaths['vel'].append(os.path.join(data_fpath, 'fte_sd_const_vel', 'fte.pickle'))
        pkl_fpaths['acc'].append(os.path.join(data_fpath, 'fte_sd_const_acc', 'fte.pickle'))

    pprint(pkl_fpaths)

    # plot the relation between head speed and reprojection error
    plot_data = {
        'pos_x': [], 'pos_y': [],
        'vel_x': [], 'vel_y': [],
        'acc_x': [], 'acc_y': [],
    }
    for k, fpaths in pkl_fpaths.items():
        for fpath in fpaths:
            # load data
            with open(fpath, 'rb') as f:
                data = pickle.load(f)   # ['positions', 'x', 'dx', 'ddx', 'shutter_delay', 'reprj_errors', 'start_frame']
            start_frame = data['start_frame']
            speeds = np.array(data['dx'])   # [n_frame, n_label]
            errors = data['reprj_errors']   # {n_camera: df} [frame, marker, camera_distance, pixel_residual, pck_threshold, error_u, error_v]
            n_frame, n_camera = speeds.shape
            # add data
            for i in range(n_camera):
                for j in range(n_frame):
                    speed = np.linalg.norm(speeds[j, 0:3])
                    error_df = errors[str(i)]

                    f = start_frame + j
                    errs = error_df.query(f'frame=={f}')['pixel_residual'].tolist()
                    plot_data[f'{k}_x'] += [speed] * len(errs)
                    plot_data[f'{k}_y'] += errs

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    keys = ['pos', 'vel', 'acc']
    methods = ['FTE', 'SD-FTE (vel)', 'SD-FTE (acc)']
    # get x-labels
    xs = np.array([plot_data[f'{k}_x'] for k in keys]).flatten()
    _, bin_edges = np.histogram(xs, bins=7)
    xlabels = []
    for i in range(len(bin_edges) - 1):
        j = i + 1
        xlabels.append('${:.1f} - {:.1f}$'.format(bin_edges[i], bin_edges[j]))
    n_data = len(xlabels)
    x = np.array(range(n_data))
    # setting the margin
    margin = 0.2  # 0 <margin< 1
    total_width = 1 - margin
    # plot bar and error bar
    for idx, (k, m) in enumerate(zip(keys, methods)):
        sp = np.array(plot_data[f'{k}_x'])  # speed
        err = np.array(plot_data[f'{k}_y'])
        xs = list(x - total_width * (1 - (2*idx+1)/len(keys)) / 2)
        ys = []
        y_minerrs = []
        y_maxerrs = []
        for i in range(len(bin_edges) - 1):
            j = i + 1
            y_data = err[(bin_edges[i] <= sp) & (sp < bin_edges[j])]
            ys.append(np.median(y_data) if len(y_data) > 0 else 0)
            y_minerrs.append(min(y_data) if len(y_data) > 0 else 0)
            y_maxerrs.append(max(y_data) if len(y_data) > 0 else 0)
        ax.bar(
            xs, ys,
            yerr=[y_minerrs, y_maxerrs],
            width=total_width / len(keys),
            label='{}'.format(m)
        )
    plt.xticks(x, xlabels)
    ax.legend()
    # ax.set_title('Speed vs Reprojection Pixel Errors')
    ax.set_xlabel('Speed $s_{head}$')
    ax.set_ylabel('Reprojection Errors')
    fig.savefig(os.path.join('.', "speed_vs_pixerror.pdf"))
