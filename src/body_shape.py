import os
import sys
import json
import numpy as np
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

from lib import misc, utils, app, metric
from lib.calib import project_points_fisheye, triangulate_points_fisheye
from lib.misc import get_markers


sns.set_theme()     # apply the default theme
plt.style.use(os.path.join('/configs', 'mplstyle.yaml'))


def save_error_dists(pix_errors, output_dir: str) -> float:
    # variables
    errors = []
    for k, df in pix_errors.items():
        errors += df['pixel_residual'].tolist()
    distances = []
    for k, df in pix_errors.items():
        distances += df['camera_distance'].tolist()

    # plot the error histogram
    xlabel = 'error (pix)'
    ylabel = 'freq'

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(errors)
    ax.set_title('Overall pixel errors (N={})'.format(len(errors)))
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


def fte(DATA_DIR, points_2d_df, mode, camera_params, start_frame, end_frame, dlc_thresh, scene_fpath, params: Dict = {}, plot: bool = False) -> str:
    # === INITIAL VARIABLES ===
    # dirs
    OUT_DIR = os.path.join(DATA_DIR, 'fte')
    os.makedirs(OUT_DIR, exist_ok=True)
    app.start_logging(os.path.join(OUT_DIR, 'fte.log'))
    # cost function
    redesc_a = 3
    redesc_b = 10
    redesc_c = 20
    # PLOT OF REDESCENDING, ABSOLUTE AND QUADRATIC COST FUNCTIONS
    # we use a redescending cost to stop outliers affecting the optimisation negatively
    if plot:
        r_x = np.arange(-20, 20, 1e-1)
        r_y1 = [misc.redescending_loss(i, redesc_a, redesc_b, redesc_c) for i in r_x]
        r_y2 = abs(r_x)
        r_y3 = r_x ** 2
        plt.figure()
        plt.plot(r_x,r_y1, label='Redescending')
        plt.plot(r_x,r_y2, label='Absolute (linear)')
        plt.plot(r_x,r_y3, label='Quadratic')
        ax = plt.gca()
        ax.set_ylim((-5, 50))
        ax.legend()
        plt.show(block=True)

    # symbolic vars
    idx       = misc.get_pose_params(mode=mode)
    sym_list  = sp.symbols(list(idx.keys()))
    positions = misc.get_3d_marker_coords(sym_list, mode=mode)

    t0 = time()

    # ========= PROJECTION FUNCTIONS ========
    def pt3d_to_2d(x, y, z, K, D, R, t):
        x_2d = x*R[0,0] + y*R[0,1] + z*R[0,2] + t.flatten()[0]
        y_2d = x*R[1,0] + y*R[1,1] + z*R[1,2] + t.flatten()[1]
        z_2d = x*R[2,0] + y*R[2,1] + z*R[2,2] + t.flatten()[2]
        # project onto camera plane
        a    = x_2d/z_2d
        b    = y_2d/z_2d
        # fisheye params
        r    = (a**2 + b**2 +1e-12)**0.5
        th   = pyo.atan(r)
        # distortion
        th_D = th * (1 + D[0]*th**2 + D[1]*th**4 + D[2]*th**6 + D[3]*th**8)
        x_P  = a*th_D/r
        y_P  = b*th_D/r
        u    = K[0,0]*x_P + K[0,2]
        v    = K[1,1]*y_P + K[1,2]
        return u, v

    def pt3d_to_x2d(x, y, z, K, D, R, t):
        u = pt3d_to_2d(x, y, z, K, D, R, t)[0]
        return u

    def pt3d_to_y2d(x, y, z, K, D, R, t):
        v = pt3d_to_2d(x, y, z, K, D, R, t)[1]
        return v

    # ========= IMPORT CAMERA & SCENE PARAMS ========
    K_arr, D_arr, R_arr, t_arr, cam_res, n_cams = camera_params
    D_arr = D_arr.reshape((-1,4))

    # ========= IMPORT DATA ========
    markers = misc.get_markers(mode=mode)
    R = 3   # measurement standard deviation (default: 5)
    _Q = [  # model parameters variance
        4, 7, 5,    # head position in inertial
        13, 9, 26,  # head rotation in inertial
        12, 32, 18, 12, # neck
        43,         # front torso
        10, 53, 34, # back torso
        90, 43,     # tail_base
        118, 51,    # tail_mid
        247, 186,   # l_shoulder, l_front_knee
        194, 164,   # r_shoulder, r_front_knee
        295, 243,   # l_hip, l_back_knee
        334, 149,   # r_hip, r_back_knee
        4, 7, 5,    # lure position in inertial
    ][:len(sym_list)]
    Q = np.array(_Q, dtype=np.float64)**2

    proj_funcs = [pt3d_to_x2d, pt3d_to_y2d]

    # save parameters
    params['start_frame'] = start_frame
    params['end_frame'] = end_frame
    params['dlc_thresh'] = dlc_thresh
    params['redesc_a'] = redesc_a
    params['redesc_b'] = redesc_b
    params['redesc_c'] = redesc_c
    params['scene_fpath'] = scene_fpath
    # params['camera_params'] = camera_params
    params['R'] = R
    params['Q'] = _Q
    with open(os.path.join(OUT_DIR, 'reconstruction_params.json'), 'w') as f:
        json.dump(params, f)

    #===================================================
    #                   Load in data
    #===================================================
    print('Generating pairwise 3D points')
    points_3d_df = utils.get_pairwise_3d_points_from_df(
        points_2d_df.query(f'likelihood > {dlc_thresh}'),
        K_arr, D_arr, R_arr, t_arr,
        triangulate_points_fisheye
    )

    #===================================================
    #                   Optimisation
    #===================================================
    print('Initialising params & variables')
    m = pyo.ConcreteModel(name='Cheetah from measurements')

    # ===== SETS =====
    N  = end_frame - start_frame + 1    # number of timesteps in trajectory
    P  = len(sym_list)  # number of pose parameters
    L  = len(markers)   # number of dlc labels per frame
    C  = n_cams         # number of cameras
    D2 = 2  # dimensionality of measurements (image points)
    D3 = 3  # dimensionality of measurements (3d points)

    m.Ts = 1.0 / fps # timestep
    m.N  = pyo.RangeSet(N)
    m.P  = pyo.RangeSet(P)
    m.L  = pyo.RangeSet(L)
    m.C  = pyo.RangeSet(C)
    m.D2 = pyo.RangeSet(D2)
    m.D3 = pyo.RangeSet(D3)

    # ======= WEIGHTS =======
    def get_meas_from_df(n, c, l, d):
        n_mask = points_2d_df['frame']  == n-1
        l_mask = points_2d_df['marker'] == markers[l-1]
        c_mask = points_2d_df['camera'] == c-1
        d_idx  = {1:'x', 2:'y'}
        val    = points_2d_df[n_mask & l_mask & c_mask]
        return val[d_idx[d]].values[0]

    def get_likelihood_from_df(n, c, l):
        n_mask = points_2d_df['frame']  == n-1
        l_mask = points_2d_df['marker'] == markers[l-1]
        c_mask = points_2d_df['camera'] == c-1
        val    = points_2d_df[n_mask & l_mask & c_mask]
        return val['likelihood'].values[0]

    def init_meas_weights(model, n, c, l):
        likelihood = get_likelihood_from_df(n + start_frame, c, l)
        if likelihood > dlc_thresh:
            return 1 / R
        else:
            return 0

    def init_model_weights(m, p):
        return 1 / Q[p-1]

    def init_measurements_df(m, n, c, l, d2):
        return get_meas_from_df(n+start_frame, c, l, d2)

    m.meas_err_weight  = pyo.Param(m.N, m.C, m.L, initialize=init_meas_weights, mutable=True)
    m.model_err_weight = pyo.Param(m.P, initialize=init_model_weights)
    m.meas             = pyo.Param(m.N, m.C, m.L, m.D2, initialize=init_measurements_df)

    # ===== MODEL VARIABLES =====
    m.x           = pyo.Var(m.N, m.P) # position
    m.dx          = pyo.Var(m.N, m.P) # velocity
    m.ddx         = pyo.Var(m.N, m.P) # acceleration
    m.poses       = pyo.Var(m.N, m.L, m.D3)
    m.slack_model = pyo.Var(m.N, m.P)
    m.slack_meas  = pyo.Var(m.N, m.C, m.L, m.D2, initialize=0.0)

    # ========= LAMBDIFY SYMBOLIC FUNCTIONS ========
    func_map = {
        'sin': pyo.sin,
        'cos': pyo.cos,
        'ImmutableDenseMatrix': np.array
    }
    # pose_to_3d = sp.lambdify(sym_list, positions, modules=[func_map])   # this is not used
    pos_funcs  = []
    for i in range(positions.shape[0]):
        lamb = sp.lambdify(sym_list, positions[i,:], modules=[func_map])
        pos_funcs.append(lamb)

    # ===== VARIABLES INITIALIZATION =====

    # estimate initial points
    frame_est = np.arange(end_frame + 1)
    print('frame_est:', frame_est.shape)
    init_x    = np.zeros((N, P))
    init_dx   = np.zeros((N, P))
    init_ddx  = np.zeros((N, P))

    nose_pts = points_3d_df[points_3d_df['marker']=='nose'][['frame', 'x', 'y', 'z']].values
    x_slope, x_intercept, *_ = linregress(nose_pts[:,0], nose_pts[:,1])
    y_slope, y_intercept, *_ = linregress(nose_pts[:,0], nose_pts[:,2])
    z_slope, z_intercept, *_ = linregress(nose_pts[:,0], nose_pts[:,3])

    x_est   = frame_est*x_slope + x_intercept
    y_est   = frame_est*y_slope + y_intercept
    z_est   = frame_est*z_slope + z_intercept
    psi_est = np.arctan2(y_slope, x_slope)

    print('idx[x_0]:', idx['x_0'])
    print('init_x:', init_x.shape)
    print('x_est:', x_est.shape)
    print('start_frame:', start_frame)
    print('end_frame:', end_frame)
    init_x[:, idx['x_0']]   = x_est[start_frame:end_frame+1]
    init_x[:, idx['y_0']]   = y_est[start_frame:end_frame+1]
    init_x[:, idx['z_0']]   = z_est[start_frame:end_frame+1]
    init_x[:, idx['psi_0']] = psi_est # psi = yaw

    for n in m.N:
        for p in m.P:
            m.x[n,p].value   = init_x[n-1,p-1]
            m.dx[n,p].value  = init_dx[n-1,p-1]
            m.ddx[n,p].value = init_ddx[n-1,p-1]
            # to init using last known value, use m.x[n,p].value = init_x[-1,p-1]
        # init pose
        var_list = [m.x[n,p].value for p in m.P]
        for l in m.L:
            [pos] = pos_funcs[l-1](*var_list)
            for d3 in m.D3:
                m.poses[n,l,d3].value = pos[d3-1]

    # ===== CONSTRAINTS =====
    print('Defining constraints')

    # NOTE: 1 based indexing for pyomo!!!!...@#^!@#&
    for state in idx:
        idx[state] += 1

    #===== POSE CONSTRAINTS =====
    print('- Pose')

    def pose_constraint(m, n, l, d3):
        var_list = [m.x[n,p] for p in m.P]
        [pos] = pos_funcs[l-1](*var_list)   # get 3d points
        return pos[d3-1] == m.poses[n,l,d3]

    m.pose_constraint = pyo.Constraint(m.N, m.L, m.D3, rule=pose_constraint)

    # define these constraint functions in a loop?
    # head
    def head_phi_0(m, n):
        return abs(m.x[n,idx['phi_0']]) <= np.pi / 6
    def head_theta_0(m, n):
        return abs(m.x[n,idx['theta_0']]) <= np.pi / 6
    # neck
    def neck_phi_1(m, n):
        return abs(m.x[n,idx['phi_1']]) <= np.pi / 6
    def neck_theta_1(m, n):
        return abs(m.x[n,idx['theta_1']]) <= np.pi / 6
    def neck_psi_1(m, n):
        return abs(m.x[n,idx['psi_1']]) <= np.pi / 6
    # front torso
    def front_torso_theta_2(m,n):
        return abs(m.x[n,idx['theta_2']]) <= np.pi / 6
    # back torso
    def back_torso_theta_3(m,n):
        return abs(m.x[n,idx['theta_3']]) <= np.pi / 6
    def back_torso_phi_3(m,n):
        return abs(m.x[n,idx['phi_3']]) <= np.pi / 6
    def back_torso_psi_3(m,n):
        return abs(m.x[n,idx['psi_3']]) <= np.pi / 6
    # tail base
    def tail_base_theta_4(m,n):
        return abs(m.x[n,idx['theta_4']]) <= np.pi / 1.5
    def tail_base_psi_4(m,n):
        return abs(m.x[n,idx['psi_4']]) <= np.pi / 1.5
    # tail mid
    def tail_mid_theta_5(m,n):
        return abs(m.x[n,idx['theta_5']]) <= np.pi / 1.5
    def tail_mid_psi_5(m,n):
        return abs(m.x[n,idx['psi_5']]) <= np.pi / 1.5
    # front left leg
    def l_shoulder_theta_6(m,n):
        return abs(m.x[n,idx['theta_6']]) <= np.pi / 2
    def l_front_knee_theta_7(m,n):
        return abs(m.x[n,idx['theta_7']] + np.pi/2) <= np.pi / 2
    # front right leg
    def r_shoulder_theta_8(m,n):
        return abs(m.x[n,idx['theta_8']]) <= np.pi / 2
    def r_front_knee_theta_9(m,n):
        return abs(m.x[n,idx['theta_9']] + np.pi/2) <= np.pi / 2
    # back left leg
    def l_hip_theta_10(m,n):
        return abs(m.x[n,idx['theta_10']]) <= np.pi / 2
    def l_back_knee_theta_11(m,n):
        return abs(m.x[n,idx['theta_11']] - np.pi/2) <= np.pi / 2
    # back right leg
    def r_hip_theta_12(m,n):
        return abs(m.x[n,idx['theta_12']]) <= np.pi / 2
    def r_back_knee_theta_13(m,n):
        return abs(m.x[n,idx['theta_13']] - np.pi/2) <= np.pi / 2

    if mode == 'default':
        m.head_phi_0           = pyo.Constraint(m.N, rule=head_phi_0)
        m.head_theta_0         = pyo.Constraint(m.N, rule=head_theta_0)
        m.neck_phi_1           = pyo.Constraint(m.N, rule=neck_phi_1)
        m.neck_theta_1         = pyo.Constraint(m.N, rule=neck_theta_1)
        m.neck_psi_1           = pyo.Constraint(m.N, rule=neck_psi_1)
        m.front_torso_theta_2  = pyo.Constraint(m.N, rule=front_torso_theta_2)
        m.back_torso_theta_3   = pyo.Constraint(m.N, rule=back_torso_theta_3)
        m.back_torso_phi_3     = pyo.Constraint(m.N, rule=back_torso_phi_3)
        m.back_torso_psi_3     = pyo.Constraint(m.N, rule=back_torso_psi_3)
        m.tail_base_theta_4    = pyo.Constraint(m.N, rule=tail_base_theta_4)
        m.tail_base_psi_4      = pyo.Constraint(m.N, rule=tail_base_psi_4)
        m.tail_mid_theta_5     = pyo.Constraint(m.N, rule=tail_mid_theta_5)
        m.tail_mid_psi_5       = pyo.Constraint(m.N, rule=tail_mid_psi_5)
        m.l_shoulder_theta_6   = pyo.Constraint(m.N, rule=l_shoulder_theta_6)
        m.l_front_knee_theta_7 = pyo.Constraint(m.N, rule=l_front_knee_theta_7)
        m.r_shoulder_theta_8   = pyo.Constraint(m.N, rule=r_shoulder_theta_8)
        m.r_front_knee_theta_9 = pyo.Constraint(m.N, rule=r_front_knee_theta_9)
        m.l_hip_theta_10       = pyo.Constraint(m.N, rule=l_hip_theta_10)
        m.l_back_knee_theta_11 = pyo.Constraint(m.N, rule=l_back_knee_theta_11)
        m.r_hip_theta_12       = pyo.Constraint(m.N, rule=r_hip_theta_12)
        m.r_back_knee_theta_13 = pyo.Constraint(m.N, rule=r_back_knee_theta_13)
    elif mode == 'head':
        m.head_phi_0           = pyo.Constraint(m.N, rule=head_phi_0)
        m.head_theta_0         = pyo.Constraint(m.N, rule=head_theta_0)

    # ===== MEASUREMENT CONSTRAINTS =====
    print('- Measurement')

    def measurement_constraints(m, n, c, l, d2):
        # project
        K, D, R, t = K_arr[c-1], D_arr[c-1], R_arr[c-1], t_arr[c-1]
        x, y, z    = m.poses[n,l,idx['x_0']], m.poses[n,l,idx['y_0']], m.poses[n,l,idx['z_0']]
        return proj_funcs[d2-1](x, y, z, K, D, R, t) - m.meas[n, c, l, d2] - m.slack_meas[n, c, l, d2] == 0

    m.measurement = pyo.Constraint(m.N, m.C, m.L, m.D2, rule=measurement_constraints)

    # ===== INTEGRATION CONSTRAINTS =====
    print('- Numerical integration')

    def backwards_euler_pos(m,n,p):
        if n > 1:
            return m.x[n,p] == m.x[n-1,p] + m.Ts*m.dx[n,p]
        else:
            return pyo.Constraint.Skip

    def backwards_euler_vel(m,n,p):
        if n > 1:
            return m.dx[n,p] == m.dx[n-1,p] + m.Ts*m.ddx[n,p]
        else:
            return pyo.Constraint.Skip

    def constant_acc(m, n, p):
        if n > 1:
            return m.ddx[n,p] == m.ddx[n-1,p] + m.slack_model[n,p]
        else:
            return pyo.Constraint.Skip

    m.integrate_p  = pyo.Constraint(m.N, m.P, rule=backwards_euler_pos)
    m.integrate_v  = pyo.Constraint(m.N, m.P, rule=backwards_euler_vel)
    m.constant_acc = pyo.Constraint(m.N, m.P, rule=constant_acc)

    # ======= OBJECTIVE FUNCTION =======
    print('Defining objective function')

    def obj(m):
        slack_model_err, slack_meas_err = 0.0, 0.0
        for n in m.N:
            # model error
            for p in m.P:
                slack_model_err += m.model_err_weight[p] * m.slack_model[n, p] ** 2
            # measurement error
            for l in m.L:
                for c in m.C:
                    for d2 in m.D2:
                        slack_meas_err += misc.redescending_loss(
                            m.meas_err_weight[n, c, l] * m.slack_meas[n, c, l, d2],
                            redesc_a,
                            redesc_b,
                            redesc_c
                        )
        return slack_meas_err + slack_model_err

    m.obj = pyo.Objective(rule=obj)

    # run the solver
    opt = SolverFactory(
        'ipopt',
        executable='/tmp/build/bin/ipopt'
    )

    # solver options
    opt.options['tol'] = 1e-1
    opt.options['print_level']  = 5
    opt.options['max_iter']     = 10000
    opt.options['max_cpu_time'] = 10000
    opt.options['OF_print_timing_statistics'] = 'yes'
    opt.options['OF_print_frequency_iter']    = 10
    opt.options['OF_hessian_approximation']   = 'limited-memory'
    opt.options['linear_solver'] = 'ma86'

    t1 = time()
    print('\nInitialization took {0:.2f} seconds\n'.format(t1 - t0))

    t0 = time()
    results = opt.solve(m, tee=True)
    t1 = time()
    print('\nOptimization took {0:.2f} seconds\n'.format(t1 - t0))

    app.stop_logging()

    # ========= SAVE FTE RESULTS ========
    x, dx, ddx = [], [], []
    for n in m.N:
        x.append([m.x[n, p].value for p in m.P])
        dx.append([m.dx[n, p].value for p in m.P])
        ddx.append([m.ddx[n, p].value for p in m.P])
    states = dict(
        x=x,
        dx=dx,
        ddx=ddx,
    )

    # calculate residual error
    positions_3d = np.array([misc.get_3d_marker_coords(state, mode=mode) for state in states['x']])
    frames = np.arange(start_frame, end_frame+1).reshape((-1, 1))
    n_frames = len(frames)
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
    save_error_dists(pix_errors, OUT_DIR)

    # save pkl/mat and video files
    out_fpath = app.save_fte(states, mode, OUT_DIR, scene_fpath, start_frame)

    fig_fpath = os.path.join(OUT_DIR, 'fte.pdf')
    app.plot_cheetah_states(x, mode=mode, out_fpath=fig_fpath)

    return out_fpath


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
        coords_3d = misc.get_3d_marker_coords(x, mode=marker_mode)
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
    positions_3d = np.array([misc.get_3d_marker_coords(state, mode=marker_mode) for state in states['smoothed_x']])
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


def tri(DATA_DIR, points_2d_df, start_frame, end_frame, dlc_thresh, camera_params, scene_fpath, params: Dict = {}) -> str:
    OUT_DIR = os.path.join(DATA_DIR, 'tri')
    os.makedirs(OUT_DIR, exist_ok=True)
    markers = misc.get_markers(mode='all')
    k_arr, d_arr, r_arr, t_arr, _, _ = camera_params

    # save reconstruction parameters
    params['start_frame'] = start_frame
    params['end_frame'] = end_frame
    params['dlc_thresh'] = dlc_thresh
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

    out_fpath = app.save_tri(positions, OUT_DIR, scene_fpath, markers, start_frame, pix_errors)

    return out_fpath


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
    k_arr, d_arr, r_arr, t_arr, cam_res, n_cams, scene_fpath = utils.find_scene_file(DATA_DIR, verbose=False)
    assert res == cam_res
    camera_params = (k_arr, d_arr, r_arr, t_arr, cam_res, n_cams)
    # load DLC data
    dlc_points_fpaths = sorted(glob(os.path.join(DLC_DIR, '*.h5')))
    assert n_cams == len(dlc_points_fpaths), f'# of dlc .h5 files != # of cams in {n_cams}_cam_scene_sba.json'

    # load measurement dataframe (pixels, likelihood)
    points_2d_df = utils.load_dlc_points_as_df(dlc_points_fpaths, frame_shifts=[0,0,0,0,0,0], verbose=False)
    filtered_points_2d_df = points_2d_df.query(f'likelihood > {args.dlc_thresh}')    # ignore points with low likelihood

    # getting parameters
    if args.end_frame == -1:
        # Automatically set start and end frame
        # defining the first and end frame as detecting all the markers on any of cameras simultaneously
        target_markers = misc.get_markers(mode)
        key_markers = ['nose', 'r_eye', 'l_eye']

        def frame_condition(i: int, n_markers: int) -> bool:
            markers_condition = ' or '.join([f'marker=="{ref}"' for ref in target_markers])
            num_marker = lambda i: len(filtered_points_2d_df.query(f'frame == {i} and ({markers_condition})')['marker'].unique())
            return num_marker(i) >= n_markers

        def frame_condition_with_key_markers(i: int, key_markers: List[str], n_min_cam: int, n_markers: int) -> bool:
            markers_condition = ' or '.join([f'marker=="{ref}"' for ref in target_markers])
            markers = filtered_points_2d_df.query(
                f'frame == {i} and ({markers_condition})'
            )['marker']

            values, counts = np.unique(markers, return_counts=True)
            if len(values) != n_markers:
                return False

            counts = [counts[np.where(values==k)[0][0]] for k in key_markers]

            return min(counts) >= n_min_cam

        start_frame, end_frame = None, None
        max_idx = int(filtered_points_2d_df['frame'].max() + 1)
        for i in range(max_idx):
            if frame_condition_with_key_markers(i, key_markers, 2, len(target_markers)):
            # if frame_condition(i, len(target_markers)):
                start_frame = i
                break
        for i in range(max_idx, 0, -1):
            if frame_condition_with_key_markers(i, key_markers, 2, len(target_markers)):
            # if frame_condition(i, len(target_markers)):
                end_frame = i
                break
        if start_frame is None or end_frame is None:
            raise('Setting frames failed. Please define start and end frames manually.')
    else:
        # User-defined frames
        start_frame = args.start_frame - 1  # 0 based indexing
        end_frame = args.end_frame % num_frames + 1 if args.end_frame == -1 else args.end_frame
    assert len(k_arr) == points_2d_df['camera'].nunique()

    print('========== Triangulation ==========\n')
    tri(DATA_DIR, points_2d_df, 0, num_frames - 1, args.dlc_thresh, camera_params, scene_fpath, params=vid_params)
    plt.close('all')
    print('========== SBA ==========\n')
    sba(DATA_DIR, points_2d_df, start_frame, end_frame, args.dlc_thresh, camera_params, scene_fpath, params=vid_params, plot=args.plot)
    plt.close('all')
    print('========== EKF ==========\n')
    ekf(DATA_DIR, points_2d_df, mode, camera_params, start_frame, end_frame, args.dlc_thresh, scene_fpath, params=vid_params)
    plt.close('all')
    print('========== FTE ==========\n')
    fte(DATA_DIR, points_2d_df, mode, camera_params, start_frame, end_frame, args.dlc_thresh, scene_fpath, params=vid_params, plot=args.plot)
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