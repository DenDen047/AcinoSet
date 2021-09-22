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

from .metrics import save_error_dists


def ekf(DATA_DIR, points_2d_df, marker_mode, camera_params, start_frame, end_frame, dlc_thresh, scene_fpath, params: Dict = {}) -> str:
    # ========= INIT VARS ========
    # dirs
    OUT_DIR = os.path.join(DATA_DIR, 'ekf')
    os.makedirs(OUT_DIR, exist_ok=True)
    # logging
    app.start_logging(os.path.join(OUT_DIR, 'ekf.log'))
    # camera
    k_arr, d_arr, r_arr, t_arr, cam_res, n_cams = camera_params
    camera_matrix = [[K, D, R, T] for K, D, R, T in zip(k_arr, d_arr, r_arr, t_arr)]
    # marker
    markers = misc.get_markers(mode=marker_mode)    # define DLC labels
    n_markers = len(markers)
    # pose
    idx = misc.get_pose_params(mode=marker_mode)    # define the indices for the states
    n_pose_params = len(idx)
    n_angular_pose_params = len([k for k in idx.keys() if 'phi' in k or 'theta' in k or 'psi' in k])
    n_states = 3 * n_pose_params
    vel_idx = n_states // 3
    acc_idx = n_states * 2 // 3
    derivs = {'d'+state: vel_idx+idx[state] for state in idx}
    derivs.update({'d'+state: vel_idx+derivs[state] for state in derivs})
    idx.update(derivs)
    # other vars
    n_frames = end_frame - start_frame + 1
    sigma_bound = 3     # if measurement residual is worse than 3 sigma, set residual to 0 and rely on predicted state only
    max_pixel_err = cam_res[0]  # used in measurement covariance R
    fps = params['vid_fps']
    sT = 1.0 / fps  # timestep

    # save reconstruction parameters
    params['marker_mode'] = marker_mode
    params['start_frame'] = start_frame
    params['end_frame'] = end_frame
    params['dlc_thresh'] = dlc_thresh
    params['sigma_bound'] = sigma_bound
    with open(os.path.join(OUT_DIR, 'reconstruction_params.json'), 'w') as f:
        json.dump(params, f)

    # ========= FUNCTION DEFINITINOS ========
    def h_function(x: np.ndarray, k: np.ndarray, d: np.ndarray, r: np.ndarray, t: np.ndarray):
        """Returns a numpy array of the 2D marker pixel coordinates (shape Nx2) for a given state vector x and camera parameters k, d, r, t.
        """
        coords_3d = misc.get_3d_marker_coords({'x': x}, mode=marker_mode)
        coords_2d = project_points_fisheye(coords_3d, k, d, r, t) # Project the 3D positions to 2D
        return coords_2d

    def predict_next_state(x: np.ndarray, dt: np.float32):
        """Returns a numpy array of the predicted states for a given state vector x and time delta dt.
        """
        acc_prediction = x[acc_idx:]
        vel_prediction = x[vel_idx:acc_idx] + dt*acc_prediction
        pos_prediction = x[:vel_idx] + dt*vel_prediction + (0.5*dt**2)*acc_prediction
        return np.concatenate([pos_prediction, vel_prediction, acc_prediction]).astype(np.float32)

    def numerical_jacobian(func, x: np.ndarray, *args):
        """Returns a numerically approximated jacobian of func with respect to x.
        Additional parameters will be passed to func using *args in the format: func(*x, *args)
        """
        n = len(x)
        eps = 1e-3

        fx = func(x, *args).flatten()
        xpeturb = x.copy()
        jac = np.empty((len(fx), n))
        for i in range(n):
            xpeturb[i] = xpeturb[i] + eps
            jac[:,i] = (func(xpeturb, *args).flatten() - fx) / eps
            xpeturb[i] = x[i]

        return jac

    # ========= LOAD DLC DATA ========

    # Load DLC 2D point files (.h5 outputs)
    points_3d_df = utils.get_pairwise_3d_points_from_df(
        points_2d_df,
        k_arr, d_arr.reshape((-1, 4)), r_arr, t_arr,
        triangulate_points_fisheye
    )

    # Restructure dataframe
    points_df = points_2d_df.set_index(['frame', 'camera', 'marker'])
    points_df = points_df.stack().unstack(level=1).unstack(level=1).unstack()

    # Pixels array
    pixels_df = points_df.loc[:, (range(n_cams), markers, ['x','y'])]
    pixels_df = pixels_df.reindex(columns=pd.MultiIndex.from_product([range(n_cams), markers, ['x','y']]))
    pixels_arr = pixels_df.to_numpy()   # (n_frames, n_cams * n_markers * 2)

    # Likelihood array
    likelihood_df = points_df.loc[:, (range(n_cams), markers, 'likelihood')]
    likelihood_df = likelihood_df.reindex(columns=pd.MultiIndex.from_product([range(n_cams), markers, ['likelihood']]))
    likelihood_arr = likelihood_df.to_numpy()   # (n_frames, n_cams * n_markers * 1)

    # ========= INITIALIZE EKF MATRICES ========

    # estimate initial points
    states = np.zeros(n_states)

    if 'lure' in markers:
        try:
            lure_pts = points_3d_df[points_3d_df['marker']=='lure'][['frame', 'x', 'y', 'z']].values
            lure_x_slope, lure_x_intercept, *_ = linregress(lure_pts[:,0], lure_pts[:,1])
            lure_y_slope, lure_y_intercept, *_ = linregress(lure_pts[:,0], lure_pts[:,2])

            lure_x_est = start_frame*lure_x_slope + lure_x_intercept # initial lure x
            lure_y_est = start_frame*lure_y_slope + lure_y_intercept # initial lure y

            states[[idx['x_l'], idx['y_l']]] = [lure_x_est, lure_y_est]             # lure x & y in inertial
            states[[idx['dx_l'], idx['dy_l']]] = [lure_x_slope/sT, lure_y_slope/sT] # lure x & y velocity in inertial
        except ValueError as e: # for when there is no lure data
            print(f'Lure initialisation error: {e} -> Lure states initialised to zero')

    points_3d_df = points_3d_df[points_3d_df['frame'].between(start_frame, end_frame)]

    nose_pts = points_3d_df[points_3d_df['marker']=='nose'][['frame', 'x', 'y', 'z']].values
    nose_x_slope, nose_x_intercept, *_ = linregress(nose_pts[:,0], nose_pts[:,1])
    nose_y_slope, nose_y_intercept, *_ = linregress(nose_pts[:,0], nose_pts[:,2])

    nose_x_est = start_frame*nose_x_slope + nose_x_intercept # initial nose x
    nose_y_est = start_frame*nose_y_slope + nose_y_intercept # initial nose y
    nose_psi_est = np.arctan2(nose_y_slope, nose_x_slope)    # initial yaw angle relative to inertial

    # INITIAL STATES
    states[[idx['x_0'], idx['y_0'], idx['psi_0']]] = [nose_x_est, nose_y_est, nose_psi_est] # head x, y & psi (yaw) in inertial
    states[[idx['dx_0'], idx['dy_0']]] = [nose_x_slope/sT, nose_y_slope/sT]                # head x & y velocity in inertial

    # INITIAL STATE COVARIANCE P - how much do we trust the initial states
    # position
    p_lin_pos = np.ones(3) * 3**2     # Know initial position within 4
    neck_len = np.ones(1) * (-0.28)
    p_ang_pos = np.ones(n_angular_pose_params) * (np.pi/4)**2   # Know initial angles within 60 degrees, heading may need to change
    p_lure_pos = np.ones(3) * 3**2
    # velocity
    p_lin_vel = np.ones(3) * 5**2     # Know this within 2.5m/s and it's a uniform random variable
    neck_vel = np.ones(1) * 0.0
    p_ang_vel = np.ones(n_angular_pose_params) * 3**2
    p_lure_vel = np.ones(3) * 5**2
    # acceleration
    p_lin_acc = np.ones(3) * 3**2
    neck_acc = np.ones(1) * 0.0
    p_ang_acc = np.ones(n_angular_pose_params) * 3**2
    p_ang_acc[10:] = 5**2
    p_lure_acc = np.ones(3) * 3**2

    if marker_mode == 'default':
        P = np.diag(np.concatenate([
            p_lin_pos, p_ang_pos[:3], neck_len, p_ang_pos[3:], p_lure_pos,
            p_lin_vel, p_ang_vel[:3], neck_vel, p_ang_vel[3:], p_lure_vel,
            p_lin_acc, p_ang_acc[:3], neck_acc, p_ang_acc[3:], p_lure_acc,
        ]))
    elif marker_mode == 'head':
        P = np.diag(np.concatenate([
            p_lin_pos, p_ang_pos[:3], p_ang_pos[3:],
            p_lin_vel, p_ang_vel[:3], p_ang_vel[3:],
            p_lin_acc, p_ang_acc[:3], p_ang_acc[3:],
        ]))

    # PROCESS COVARIANCE Q - how 'noisy' the constant acceleration model is
    qb_list = [
        5.0, 5.0, 5.0, 10.0, 10.0, 10.0,    # head x, y, z, phi, theta, psi in inertial
        5.0, 5.0, 25.0, 5.0,   # neck length, phi, theta, psi
        50.0,             # front-torso theta
        5.0, 50.0, 25.0,  # back torso phi, theta, psi
        100.0, 30.0,      # tail base theta, psi
        140.0, 40.0,      # tail mid theta, psi
        350.0, 200.0,     # l_shoulder theta, l_front_knee theta
        350.0, 200.0,     # r_shoulder theta, r_front_knee theta
        450.0, 400.0,     # l_hip theta, l_back_knee theta
        450.0, 400.0,     # r_hip theta, r_back_knee theta
        5.0, 5.0, 5.0,    # lure x, y, z in inertial - same as head
    ]
    qb_list = qb_list[:n_pose_params]

    qb = (np.diag(qb_list))**2
    Q = np.block([
        [sT**4/4 * qb, sT**3/2 * qb, sT**2/2 * qb],
        [sT**3/2 * qb, sT**2 * qb, sT * qb],
        [sT**2/2 * qb, sT * qb, qb],
    ])

    # MEASUREMENT COVARIANCE R
    dlc_cov = 0     # 5**2
    cal_covs = [0.137, 0.236, 0.176, 0.298, 0.087, 0.116]
    # cal_covs = [0.5, 2.271, 1.795, 0.310, 0.087, 0.116]
    # cal_covs = [0., 0., 0., 0., 0., 0.]
    assert n_cams == len(cal_covs)

    # State prediction function jacobian F - shape: (n_states, n_states)
    rng = np.arange(n_states - vel_idx)
    rng_acc = np.arange(n_states - acc_idx)
    F = np.eye(n_states)
    F[rng, rng+vel_idx] = sT
    F[rng_acc, rng_acc+acc_idx] = sT**2/2

    # Allocate space for storing EKF data
    states_est_hist = np.zeros((n_frames, n_states))
    states_pred_hist = states_est_hist.copy()
    P_est_hist = np.zeros((n_frames, n_states, n_states))
    P_pred_hist = P_est_hist.copy()

    # ========= RUN EKF & SMOOTHER ========
    t0 = time()
    outliers_ignored = 0

    print('Running EKF...')
    for i in tqdm(range(n_frames)):
        # ========== PREDICTION ==========
        # Predict State
        states = predict_next_state(states, sT).flatten()
        states_pred_hist[i] = states

        # Projection of the state covariance
        P = F @ P @ F.T + Q
        P_pred_hist[i] = P

        # ============ UPDATE ============
        # Measurement
        H = np.zeros((n_cams*n_markers*2, n_states))
        h = np.zeros((n_cams*n_markers*2))  # same as H[:, 0].copy()
        for j in range(n_cams):
            # State measurement
            h[j*n_markers*2:(j+1)*n_markers*2] = h_function(states[:vel_idx], *camera_matrix[j]).flatten()
            # Jacobian - shape: (2*n_markers, n_states)
            H[j*n_markers*2:(j+1)*n_markers*2, 0:vel_idx] = numerical_jacobian(h_function, states[:vel_idx], *camera_matrix[j])

        # Measurement Covariance R
        likelihood = likelihood_arr[i + start_frame]
        bad_point_mask = np.repeat(likelihood<dlc_thresh, 2)
        dlc_cov_arr = []
        for cov in cal_covs:
            dlc_cov_arr += [dlc_cov + 2 * cov / min(cal_covs)] * (n_markers*2)
        dlc_cov_arr = np.array(dlc_cov_arr)
        # dlc_cov_arr = [dlc_cov + c for c in cal_covs] * (n_markers*2)
        # dlc_cov_arr = np.array(dlc_cov_arr)

        dlc_cov_arr[bad_point_mask] = max_pixel_err # change this to be independent of cam res?
        R = np.diag(dlc_cov_arr**2)

        # Residual
        z_k = pixels_arr[i + start_frame]
        residual = np.nan_to_num(z_k - h)

        # Residual Covariance S
        S = (H @ P @ H.T) + R
        temp = sigma_bound * np.sqrt(np.diag(S))    # if measurement residual is worse than 3 sigma, set residual to 0 and rely on predicted state only
        for j in range(0, len(residual), 2):
            if np.abs(residual[j])>temp[j] or np.abs(residual[j+1])>temp[j+1]:
                # residual[j:j+2] = 0
                outliers_ignored += 1

        # Kalman Gain
        K = P @ H.T @ np.linalg.inv(S)

        # Correction
        states = states + K @ residual
        states_est_hist[i] = states

        # Update State Covariance
        P = (np.eye(K.shape[0]) - K @ H) @ P
        P_est_hist[i] = P

    print('EKF complete!')

    # Run Kalman Smoother
    smooth_states_est_hist = states_est_hist.copy()
    smooth_P_est_hist = P_est_hist.copy()
    for i in range(n_frames-2, -1, -1):
        A = P_est_hist[i] @ F.T @ np.linalg.inv(P_pred_hist[i+1])
        smooth_states_est_hist[i] = states_est_hist[i] + A @ (smooth_states_est_hist[i+1] - states_pred_hist[i+1])
        smooth_P_est_hist[i] = P_est_hist[i] + A @ (smooth_P_est_hist[i+1] - P_pred_hist[i+1]) @ A.T
    print('Kalman Smoother complete!')

    opt_time = time() - t0

    app.stop_logging()

    # ========= SAVE EKF RESULTS ========
    states = dict(
        x=states_est_hist[:, :vel_idx],
        dx=states_est_hist[:, vel_idx:acc_idx],
        ddx=states_est_hist[:, acc_idx:],
        smoothed_x=smooth_states_est_hist[:, :vel_idx],
        smoothed_dx=smooth_states_est_hist[:, vel_idx:acc_idx],
        smoothed_ddx=smooth_states_est_hist[:, acc_idx:]
    )

    # calculate residual error
    positions_3d = np.array([misc.get_3d_marker_coords({'x': state}, mode=marker_mode) for state in states['smoothed_x']])
    frames = np.arange(start_frame, end_frame+1).reshape((-1, 1))
    points_3d = []
    for i, m in enumerate(markers):
        _pt3d = np.squeeze(positions_3d[:, i, :])
        marker_arr = np.array([m] * n_frames).reshape((-1, 1))
        _pt3d = np.hstack((frames, marker_arr, _pt3d))
        points_3d.append(_pt3d)
    points_3d_df = pd.DataFrame(
        np.vstack(points_3d),
        columns=['frame', 'marker', 'x', 'y', 'z'],
    ).astype(
        {'frame': 'int64', 'marker': 'str', 'x': 'float64', 'y': 'float64', 'z': 'float64'}
    )

    pix_errors = metric.residual_error(points_2d_df, points_3d_df, markers, camera_params)
    mae_all = save_error_dists(pix_errors, OUT_DIR)
    maes = []
    for k, df in pix_errors.items():
        maes.append(round(df['pixel_residual'].mean(), 3))

    print(f'\tOutliers ignored: {outliers_ignored}')
    print('\tOptimization took {0:.2f} seconds'.format(opt_time))
    print('\tReprojection MAEs: {}'.format(maes))
    print('\tReprojection MAE: {:.3f} pix'.format(mae_all))

    # save the videos
    out_fpath = app.save_ekf(states, marker_mode, OUT_DIR, scene_fpath, start_frame, save_videos=True)

    fig_fpath = os.path.join(OUT_DIR, 'ekf.pdf')
    app.plot_cheetah_states(states['x'], states['smoothed_x'], marker_mode, fig_fpath)

    return out_fpath
