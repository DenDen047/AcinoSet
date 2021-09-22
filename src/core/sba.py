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


def sba(DATA_DIR, points_2d_df, start_frame, end_frame, dlc_thresh, camera_params, scene_fpath, params: Dict = {}, plot: bool = False) -> str:
    OUT_DIR = os.path.join(DATA_DIR, 'sba')
    os.makedirs(OUT_DIR, exist_ok=True)
    app.start_logging(os.path.join(OUT_DIR, 'sba.log'))
    markers = misc.get_markers()

    # save reconstruction parameters
    params['start_frame'] = start_frame
    params['end_frame'] = end_frame
    params['dlc_thresh'] = dlc_thresh
    with open(os.path.join(OUT_DIR, 'reconstruction_params.json'), 'w') as f:
        json.dump(params, f)

    # get 3D points
    points_2d_df = points_2d_df.query(f'likelihood > {dlc_thresh}')
    points_2d_df = points_2d_df[points_2d_df['frame'].between(start_frame, end_frame)]
    points_3d_df, residuals = app.sba_points_fisheye(scene_fpath, points_2d_df)

    app.stop_logging()
    if plot:
        plt.plot(residuals['before'], label='Cost before')
        plt.plot(residuals['after'], label='Cost after')
        plt.legend()
        fig_fpath = os.path.join(OUT_DIR, 'sba.pdf')
        plt.savefig(fig_fpath, transparent=True)
        print(f'Saved {fig_fpath}\n')
        plt.show(block=False)

    # calculate residual error
    pix_errors = metric.residual_error(points_2d_df, points_3d_df, markers, camera_params)
    save_error_dists(pix_errors, OUT_DIR)


    # ========= SAVE SBA RESULTS ========
    positions = np.full((end_frame - start_frame + 1, len(markers), 3), np.nan)

    for i, marker in enumerate(markers):
        marker_pts = points_3d_df[points_3d_df['marker']==marker][['frame', 'x', 'y', 'z']].values
        for frame, *pt_3d in marker_pts:
            positions[int(frame)-start_frame, i] = pt_3d

    out_fpath = app.save_sba(positions, OUT_DIR, scene_fpath, markers, start_frame)

    return out_fpath
