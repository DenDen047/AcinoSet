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
    ax.set_title('Overall pixel errors (N={}, median={:.3f})'.format(len(errors), np.median(errors)))
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


def fte(DATA_DIR, points_2d_df, mode, camera_params, start_frame, end_frame, dlc_thresh, scene_fpath, params: Dict = {}, shutter_delay: bool = False, interpolation_mode: str = 'pos', plot: bool = False) -> str:
    # === INITIAL VARIABLES ===
    sd = shutter_delay
    intermode = interpolation_mode
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
    sym_list  = sp.symbols(list(idx.keys()))    # [x_0, y_0, z_0, phi_0, theta_0, psi_0]
    positions = misc.get_3d_marker_coords({'x': sym_list}, mode=mode)

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
    _Q = {  # model parameters variance
        'x_0': 4,
        'y_0': 7,
        'z_0': 5,
        'phi_0': 13,
        'theta_0': 9,
        'psi_0': 26,
        'l_1': 32,
        'phi_1': 32,
        'theta_1': 18,
        'psi_1': 12,
        'theta_2': 43,
        'phi_3': 10,
        'theta_3': 53,
        'psi_3': 34,
        'theta_4': 90,
        'psi_4': 43,
        'theta_5': 118,
        'psi_5': 51,
        'theta_6': 247,
        'theta_7': 186,
        'theta_8': 194,
        'theta_9': 164,
        'theta_10': 295,
        'theta_11': 243,
        'theta_12': 334,
        'theta_13': 149,
        'x_l': 4,
        'y_l': 7,
        'z_l': 5,
    }
    Q = np.array([_Q[str(s)] for s in sym_list], dtype=np.float64)**2

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
    print('----- Generating pairwise 3D points -----')
    points_3d_df = utils.get_pairwise_3d_points_from_df(
        points_2d_df.query(f'likelihood > {dlc_thresh}'),
        K_arr, D_arr, R_arr, t_arr,
        triangulate_points_fisheye
    )

    #===================================================
    #                   Optimisation
    #===================================================
    print('----- Initialising params & Variables -----')
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
    m.shutter_delay = pyo.Var(m.N, m.C, initialize=0.0)

    # ========= LAMBDIFY SYMBOLIC FUNCTIONS ========
    func_map = {
        'sin': pyo.sin,
        'cos': pyo.cos,
        'ImmutableDenseMatrix': np.array
    }
    pos_funcs  = []
    for i in range(positions.shape[0]):
        lamb = sp.lambdify(sym_list, positions[i,:], modules=[func_map])
        pos_funcs.append(lamb)

    # ===== VARIABLES INITIALIZATION =====
    print('Parameters')

    # estimate initial points
    frame_est = np.arange(end_frame + 1)
    print('- frame_est:', frame_est.shape)
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

    print('- idx[x_0]:', idx['x_0'])
    print('- init_x:', init_x.shape)
    print('- x_est:', x_est.shape)
    print('- start_frame:', start_frame)
    print('- end_frame:', end_frame)
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

    #===== SHUTTER DELAY CONSTRAINTS =====
    print('- Shutter delay')

    def shutter_base_constraint(m, n):
        return m.shutter_delay[n, 1] == 0.0

    def shutter_delay_constraint(m, n, c):
        return (-m.Ts, m.shutter_delay[n, c], m.Ts)

    def disable_shutter_delay(m, n, c):
        return m.shutter_delay[n, c] == 0.0

    m.shutter_base_constraint = pyo.Constraint(m.N, rule=shutter_base_constraint)
    m.shutter_delay_constraint = pyo.Constraint(m.N, m.C, rule=shutter_delay_constraint if sd else disable_shutter_delay)

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
    def front_torso_theta_2(m, n):
        return abs(m.x[n,idx['theta_2']]) <= np.pi / 6
    # back torso
    def back_torso_theta_3(m, n):
        return abs(m.x[n,idx['theta_3']]) <= np.pi / 6
    def back_torso_phi_3(m, n):
        return abs(m.x[n,idx['phi_3']]) <= np.pi / 6
    def back_torso_psi_3(m, n):
        return abs(m.x[n,idx['psi_3']]) <= np.pi / 6
    # tail base
    def tail_base_theta_4(m, n):
        return abs(m.x[n,idx['theta_4']]) <= np.pi / 1.5
    def tail_base_psi_4(m, n):
        return abs(m.x[n,idx['psi_4']]) <= np.pi / 1.5
    # tail mid
    def tail_mid_theta_5(m, n):
        return abs(m.x[n,idx['theta_5']]) <= np.pi / 1.5
    def tail_mid_psi_5(m, n):
        return abs(m.x[n,idx['psi_5']]) <= np.pi / 1.5
    # front left leg
    def l_shoulder_theta_6(m, n):
        return abs(m.x[n,idx['theta_6']]) <= np.pi / 2
    def l_front_knee_theta_7(m, n):
        return abs(m.x[n,idx['theta_7']] + np.pi/2) <= np.pi / 2
    # front right leg
    def r_shoulder_theta_8(m, n):
        return abs(m.x[n,idx['theta_8']]) <= np.pi / 2
    def r_front_knee_theta_9(m, n):
        return abs(m.x[n,idx['theta_9']] + np.pi/2) <= np.pi / 2
    # back left leg
    def l_hip_theta_10(m, n):
        return abs(m.x[n,idx['theta_10']]) <= np.pi / 2
    def l_back_knee_theta_11(m, n):
        return abs(m.x[n,idx['theta_11']] - np.pi/2) <= np.pi / 2
    # back right leg
    def r_hip_theta_12(m, n):
        return abs(m.x[n,idx['theta_12']]) <= np.pi / 2
    def r_back_knee_theta_13(m, n):
        return abs(m.x[n,idx['theta_13']] - np.pi/2) <= np.pi / 2

    if 'phi_0' in markers:
        m.head_phi_0           = pyo.Constraint(m.N, rule=head_phi_0)
    if 'theta_0' in markers:
        m.head_theta_0         = pyo.Constraint(m.N, rule=head_theta_0)
    if 'phi_1' in markers:
        m.neck_phi_1           = pyo.Constraint(m.N, rule=neck_phi_1)
    if 'theta_1' in markers:
        m.neck_theta_1         = pyo.Constraint(m.N, rule=neck_theta_1)
    if 'psi_1' in markers:
        m.neck_psi_1           = pyo.Constraint(m.N, rule=neck_psi_1)
    if 'theta_2' in markers:
        m.front_torso_theta_2  = pyo.Constraint(m.N, rule=front_torso_theta_2)
    if 'theta_3' in markers:
        m.back_torso_theta_3   = pyo.Constraint(m.N, rule=back_torso_theta_3)
    if 'phi_3' in markers:
        m.back_torso_phi_3     = pyo.Constraint(m.N, rule=back_torso_phi_3)
    if 'psi_3' in markers:
        m.back_torso_psi_3     = pyo.Constraint(m.N, rule=back_torso_psi_3)
    if 'theta_4' in markers:
        m.tail_base_theta_4    = pyo.Constraint(m.N, rule=tail_base_theta_4)
    if 'psi_4' in markers:
        m.tail_base_psi_4      = pyo.Constraint(m.N, rule=tail_base_psi_4)
    if 'theta_5' in markers:
        m.tail_mid_theta_5     = pyo.Constraint(m.N, rule=tail_mid_theta_5)
    if 'psi_5' in markers:
        m.tail_mid_psi_5       = pyo.Constraint(m.N, rule=tail_mid_psi_5)
    if 'theta_6' in markers:
        m.l_shoulder_theta_6   = pyo.Constraint(m.N, rule=l_shoulder_theta_6)
    if 'theta_7' in markers:
        m.l_front_knee_theta_7 = pyo.Constraint(m.N, rule=l_front_knee_theta_7)
    if 'theta_8' in markers:
        m.r_shoulder_theta_8   = pyo.Constraint(m.N, rule=r_shoulder_theta_8)
    if 'theta_9' in markers:
        m.r_front_knee_theta_9 = pyo.Constraint(m.N, rule=r_front_knee_theta_9)
    if 'theta_10' in markers:
        m.l_hip_theta_10       = pyo.Constraint(m.N, rule=l_hip_theta_10)
    if 'theta_11' in markers:
        m.l_back_knee_theta_11 = pyo.Constraint(m.N, rule=l_back_knee_theta_11)
    if 'theta_12' in markers:
        m.r_hip_theta_12       = pyo.Constraint(m.N, rule=r_hip_theta_12)
    if 'theta_13' in markers:
        m.r_back_knee_theta_13 = pyo.Constraint(m.N, rule=r_back_knee_theta_13)

    # ===== MEASUREMENT CONSTRAINTS =====
    print('- Measurement')

    def measurement_constraints(m, n, c, l, d2):
        # m ... model
        # n ... frame
        # c ... camera
        # l ... DLC label
        # project
        K, D, R, t = K_arr[c-1], D_arr[c-1], R_arr[c-1], t_arr[c-1]
        tau = m.shutter_delay[n, c]
        x = m.poses[n,l,1] + m.dx[n,idx['x_0']]*tau + m.ddx[n,idx['x_0']]*(tau**2)
        y = m.poses[n,l,2] + m.dx[n,idx['y_0']]*tau + m.ddx[n,idx['y_0']]*(tau**2)
        z = m.poses[n,l,3] + m.dx[n,idx['z_0']]*tau + m.ddx[n,idx['z_0']]*(tau**2)
        return proj_funcs[d2-1](x, y, z, K, D, R, t) - m.meas[n, c, l, d2] - m.slack_meas[n, c, l, d2] == 0

    m.measurement = pyo.Constraint(m.N, m.C, m.L, m.D2, rule=measurement_constraints)

    # ===== INTEGRATION CONSTRAINTS =====
    print('- Numerical integration')
    l_idx = [i for k, i in idx.items() if k[:2] == 'l_']

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
            if p in l_idx:
                return (None, m.ddx[n,p], None)
            else:
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
    opt.options['tol'] = 1e-3
    opt.options['print_level']  = 5
    opt.options['max_iter']     = 10000
    opt.options['max_cpu_time'] = 10000
    opt.options['OF_print_timing_statistics'] = 'yes'
    opt.options['OF_print_frequency_iter']    = 10
    opt.options['OF_hessian_approximation']   = 'limited-memory'
    opt.options['linear_solver'] = 'ma86'

    t1 = time()
    print('\nInitialization took {0:.2f} seconds\n'.format(t1 - t0))

    print('----- Optimization -----')
    t0 = time()
    results = opt.solve(m, tee=True)
    t1 = time()
    print('\nOptimization took {0:.2f} seconds\n'.format(t1 - t0))

    app.stop_logging()

    # ========= SAVE FTE RESULTS ========
    print('----- Shutter Delay -----')
    for c in m.C:
        result = pd.DataFrame(pd.Series([m.shutter_delay[n, c].value for n in m.N]).describe()).transpose()
        print(f'Camera {c}')
        print(result)
    x, dx, ddx = [], [], []
    for n in m.N:
        x.append([m.x[n, p].value for p in m.P])
        dx.append([m.dx[n, p].value for p in m.P])
        ddx.append([m.ddx[n, p].value for p in m.P])
    shutter_delay = [[m.shutter_delay[n,c].value for n in m.N] for c in m.C]
    states = dict(
        x=x,
        dx=dx,
        ddx=ddx,
        shutter_delay=[[m.shutter_delay[n,c].value for n in m.N] for c in m.C]
    )

    # save pkl/mat and video files
    out_fpath = app.save_fte(states, mode, OUT_DIR, scene_fpath, start_frame, directions=True, save_videos=True)

    # calculate residual error
    positions_3ds = misc.get_all_marker_coords_from_states(states, n_cams, directions=True, mode=mode)
    points_3d_dfs = []
    for positions_3d in positions_3ds:
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
        ).astype({'frame': 'int64', 'marker': 'str', 'x': 'float64', 'y': 'float64', 'z': 'float64'})
        points_3d_dfs.append(points_3d_df)
    pix_errors = metric.residual_error(points_2d_df, points_3d_dfs, markers, camera_params)
    save_error_dists(pix_errors, OUT_DIR)

    # plot cheetah state
    fig_fpath = os.path.join(OUT_DIR, 'fte.pdf')
    app.plot_cheetah_states(x, mode=mode, out_fpath=fig_fpath)
    fig_fpath = os.path.join(OUT_DIR, 'shutter_delay.pdf')
    app.plot_shutter_delay(shutter_delay, out_fpath=fig_fpath)

    return out_fpath


def dlc(DATA_DIR, OUT_DIR, dlc_thresh, params: Dict = {}) -> Dict:
    df_fpaths = sorted(glob(os.path.join(OUT_DIR, 'cam[1-9]*.h5'))) # original vids should be in the parent dir
    video_fpaths = sorted(glob(os.path.join(DATA_DIR, 'cam[1-9].mp4'))) # original vids should be in the parent dir

    # save parameters
    params['dlc_thresh'] = dlc_thresh
    with open(os.path.join(OUT_DIR, 'video_params.json'), 'w') as f:
        json.dump(params, f)

    # load dataframes
    point2d_dfs = []
    for df_fpath in df_fpaths:
        df = pd.read_hdf(df_fpath)
        point2d_dfs.append(df)

    app.create_labeled_videos(point2d_dfs, video_fpaths, out_dir=OUT_DIR, draw_skeleton=True, pcutoff=dlc_thresh, lure=False)

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

    mode = 'head_stabilize'

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
    DLC_DIR = os.path.join(DATA_DIR, 'dlc')
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
    points_2d_df = utils.load_dlc_points_as_df(dlc_points_fpaths, verbose=False)
    filtered_points_2d_df = points_2d_df.query(f'likelihood > {args.dlc_thresh}')    # ignore points with low likelihood

    # getting parameters
    if args.end_frame == -1:
        # Automatically set start and end frame
        # defining the first and end frame as detecting all the markers on any of cameras simultaneously
        target_markers = misc.get_markers(mode)

        def frame_condition(i: int, target_markers: List[str]) -> bool:
            markers_condition = ' or '.join([f'marker=="{ref}"' for ref in target_markers])
            num_marker = lambda i: len(filtered_points_2d_df.query(f'frame == {i} and ({markers_condition})')['marker'].unique())
            return num_marker(i) >= len(target_markers)

        def frame_condition_with_key_markers(i: int, key_markers: List[str], n_min_cam: int) -> bool:
            markers_condition = ' or '.join([f'marker=="{ref}"' for ref in key_markers])
            markers = filtered_points_2d_df.query(
                f'frame == {i} and ({markers_condition})'
            )['marker']

            values, counts = np.unique(markers, return_counts=True)
            if len(values) != len(key_markers):
                return False

            return min(counts) >= n_min_cam

        start_frame, end_frame = None, None
        max_idx = int(filtered_points_2d_df['frame'].max() + 1)
        for i in range(max_idx):    # start_frame
            if frame_condition_with_key_markers(i, target_markers, 2):
            # if frame_condition(i, target_markers):
                start_frame = i
                break
        for i in range(max_idx, 0, -1): # end_frame
            if frame_condition_with_key_markers(i, target_markers, 2):
            # if frame_condition(i, target_markers):
                end_frame = i
                break
        if start_frame is None or end_frame is None:
            raise('Setting frames failed. Please define start and end frames manually.')
    else:
        # User-defined frames
        start_frame = args.start_frame - 1  # 0 based indexing
        end_frame = args.end_frame % num_frames + 1 if args.end_frame == -1 else args.end_frame
    assert len(k_arr) == points_2d_df['camera'].nunique()

    print('start frame:', start_frame)
    print('end frame:', end_frame)

    print('========== FTE ==========\n')
    fte(DATA_DIR, points_2d_df, mode, camera_params, start_frame, end_frame, args.dlc_thresh, scene_fpath, params=vid_params, shutter_delay=True, interpolation_mode='acc', plot=args.plot)
    plt.close('all')
