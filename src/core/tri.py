import os
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

from lib import misc, utils, app, metric
from lib.calib import project_points_fisheye, triangulate_points_fisheye
from lib.misc import get_markers

from .metrics import save_error_dists


def tri(DATA_DIR, points_2d_df, start_frame, end_frame, dlc_thresh, camera_params, scene_fpath, params: Dict = {}) -> str:
    OUT_DIR = os.path.join(DATA_DIR, 'tri')
    os.makedirs(OUT_DIR, exist_ok=True)
    markers = misc.get_markers(mode='all')
    k_arr, d_arr, r_arr, t_arr, _, _, _ = camera_params

    # save reconstruction parameters
    params['start_frame'] = start_frame
    params['end_frame'] = end_frame
    params['dlc_thresh'] = dlc_thresh
    params['scene_fpath'] = scene_fpath
    params['markers'] = dict(zip(markers, range(len(markers))))
    params['skeletons'] = misc.get_skeleton('all')
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

    # ========= SAVE TRIANGULATION RESULTS ========
    positions = np.full((end_frame - start_frame + 1, len(markers), 3), np.nan)

    for i, marker in enumerate(markers):
        marker_pts = points_3d_df[points_3d_df['marker']==marker][['frame', 'x', 'y', 'z']].values
        for frame, *pt_3d in marker_pts:
            positions[int(frame) - start_frame, i] = pt_3d

    out_fpath = app.save_tri(positions, OUT_DIR, camera_params, markers, start_frame, pix_errors, save_videos=True)

    return out_fpath

