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
    args = parser.parse_args()

    mode = 'head'
    sd = False
    sd_mode = 'const'
    intermode = 'pos'

    data_fpahts = [
        '/data/2017_08_29/top/phantom/run1_1/',
        '/data/2019_03_05/lily/run/',
        '/data/2019_03_07/menya/run/',
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
    xs = []
    ys = []
    zs = []
    for idx, (k, m) in enumerate(zip(keys, methods)):
        sp = np.array(plot_data[f'{k}_x'])  # speed
        err = np.array(plot_data[f'{k}_y'])
        for i in range(len(bin_edges) - 1):
            j = i + 1
            y_data = list(err[(bin_edges[i] <= sp) & (sp < bin_edges[j])])
            xs += ['${:.1f} - {:.1f}$'.format(bin_edges[i], bin_edges[j])] * len(y_data)
            ys += y_data
            zs += [m] * len(y_data)
    df = pd.DataFrame({
        'range': xs,
        'error': ys,
        'method': zs,
    })
    sns.boxplot(
        x='range', y='error', data=df, hue='method',
        fliersize=1,
        ax=ax
    )
    ax.set_ylim(0, 20)
    ax.legend()
    # ax.set_title('Speed vs Reprojection Pixel Errors')
    ax.set_xlabel('Speed $s_{head}$ (m/s)', fontsize='medium')
    ax.set_ylabel('Reprojection Errors (px)', fontsize='medium')
    fig.savefig(os.path.join('.', "speed_vs_pixerror.pdf"))
