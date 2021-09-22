import os
import sys
import json
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


def save_error_dists(pix_errors, output_dir: str) -> float:
    # variables
    errors = []
    for k, df in pix_errors.items():
        errors += df['pixel_residual'].tolist()
    distances = []
    for k, df in pix_errors.items():
        distances += df['camera_distance'].tolist()

    # PCK (percentage correct keypoints)
    ratio = 0.5
    distances = []
    pck_threshold = []
    for k, df in pix_errors.items():
        distances += df["pixel_residual"].tolist()
        pck_threshold += df["pck_threshold"].tolist()
    distances = np.asarray(list(map(float, distances)))
    pck_threshold = np.asarray(list(map(float, pck_threshold)))
    pck = 100.0 * (np.sum(distances <= (ratio * pck_threshold)) / len(distances))

    # plot the error histogram
    xlabel = 'error (pix)'
    ylabel = 'freq'

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(errors)
    ax.set_title('Overall pixel errors (N={}, median={:.3f}, PCK={:.2f})'.format(len(errors), np.median(errors), pck))
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
