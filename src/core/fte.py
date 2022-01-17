import os
import sys
import json
import copy
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

from .metrics import save_error_dists


def pyo_i(i: int) -> int:
    return i + 1


def fte(
    OUT_DIR,
    points_2d_df,
    mode, camera_params,
    start_frame, end_frame, body_start_frame, body_end_frame, lure_start_frame, lure_end_frame, dlc_thresh,
    scene_fpath,
    params: Dict = {},
    lure: bool = False,
    shutter_delay: bool = False, shutter_delay_mode: str = 'const', interpolation_mode: str = 'pos',
    video: bool = True,
    plot: bool = False
) -> str:
    params['start_frame'] = start_frame
    params['end_frame'] = end_frame
    params['body_start_frame'] = body_start_frame
    params['body_end_frame'] = body_end_frame
    if lure:
        params['lure_start_frame'] = lure_start_frame
        params['lure_end_frame'] = lure_end_frame
    params['redesc_a'] = 3
    params['redesc_b'] = 10
    params['redesc_c'] = 20
    params['R'] = {
        'nose': 1.2,
        'l_eye': 1.24,
        'r_eye': 1.18,
        'neck_base': 2.08,
        'spine': 2.04,
        'tail_base': 2.52,
        'tail1': 2.73,
        'tail2': 1.83,
        'r_shoulder': 3.47,
        'r_front_knee': 2.75,
        'r_front_ankle': 2.69,
        'r_front_paw': 2.24,
        'l_shoulder': 3.4,
        'l_front_knee': 2.91,
        'l_front_ankle': 2.85,
        'l_front_paw': 2.27,
        'r_hip': 3.26,
        'r_back_knee': 2.76,
        'r_back_ankle': 2.33,
        'r_back_paw': 2.4,
        'l_hip': 3.53,
        'l_back_knee': 2.69,
        'l_back_ankle': 2.49,
        'l_back_paw': 2.34,
    }
    params['Q'] = {  # model parameters variance
        'x_0': 4,
        'y_0': 7,
        'z_0': 5,
        'phi_0': 13,
        'theta_0': 9,
        'psi_0': 26,
        'l_1': 4,
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
    params['dlc_thresh'] = dlc_thresh
    params['scene_fpath'] = scene_fpath

    body_state = _fte(
        OUT_DIR,
        points_2d_df, mode, camera_params,
        body_start_frame, body_end_frame,
        dlc_thresh,
        scene_fpath,
        params=params,
        lure=False,
        shutter_delay=shutter_delay,
        shutter_delay_mode=shutter_delay_mode,
        interpolation_mode=interpolation_mode,
        video=video,
        plot=plot
    )
    if lure:
        lure_state = _fte(
            OUT_DIR,
            points_2d_df, '', camera_params,
            lure_start_frame, lure_end_frame,
            dlc_thresh,
            scene_fpath,
            params=params,
            lure=True,
            shutter_delay=shutter_delay,
            shutter_delay_mode=shutter_delay_mode,
            interpolation_mode=interpolation_mode,
            video=video,
            plot=plot
        )
    with open(os.path.join(OUT_DIR, 'reconstruction_params.json'), 'w') as f:
        json.dump(params, f)
    pprint(params)

    # reshape with start and end frame
    if lure:
        state = {}
        bs = start_frame - body_start_frame
        be = len(body_state['x']) - (body_end_frame - end_frame)
        ls = start_frame - lure_start_frame
        le = len(lure_state['x']) - (lure_end_frame - end_frame)
        for i in ['x', 'dx', 'ddx']:
            state[i] = np.concatenate((body_state[i][bs:be, :], lure_state[i][ls:le, :]), axis=1)
        state['shutter_delay'] = body_state['shutter_delay'][bs:be, :]
    else:
        state = body_state

    # ========= SAVE FTE RESULTS ========
    K_arr, D_arr, R_arr, t_arr, cam_res, cam_names, n_cams = camera_params
    intermode = interpolation_mode
    markers = misc.get_markers(mode=mode, lure=lure)

    # calculate residual error
    positions_3ds = misc.get_all_marker_coords_from_states(state, n_cams, mode=mode, lure=lure, directions=True, intermode=intermode)
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
    state['reprj_errors'] = pix_errors
    save_error_dists(pix_errors, OUT_DIR)

    # save pkl/mat and video files
    out_fpath = app.save_fte(state, mode, OUT_DIR, camera_params, start_frame, lure=lure, directions=True, intermode=intermode, save_videos=video)

    # plot cheetah state
    fig_fpath = os.path.join(OUT_DIR, 'fte.pdf')
    app.plot_cheetah_states(state['x'], mode=mode, lure=lure, out_fpath=fig_fpath)
    if shutter_delay:
        fig_fpath = os.path.join(OUT_DIR, 'shutter_delay.pdf')
        app.plot_shutter_delay(state['shutter_delay'], out_fpath=fig_fpath)

    return out_fpath

def _fte(
    OUT_DIR,
    points_2d_df,
    mode, camera_params,
    start_frame, end_frame, dlc_thresh,
    scene_fpath,
    params: Dict = {},
    lure: bool = False,
    shutter_delay: bool = False, shutter_delay_mode: str = 'const', interpolation_mode: str = 'pos',
    video: bool = True,
    plot: bool = False
) -> Dict:
    # === INITIAL VARIABLES ===
    # options
    sd = shutter_delay
    sd_mode = shutter_delay_mode
    intermode = interpolation_mode
    if sd:
        assert sd_mode == 'const' or sd_mode == 'variable'
        assert intermode == 'vel' or intermode == 'acc'
    else:
        assert intermode == 'pos'
    # dirs
    os.makedirs(OUT_DIR, exist_ok=True)
    app.start_logging(os.path.join(OUT_DIR, 'fte.log'))

    # symbolic vars
    idx       = misc.get_pose_params(mode=mode, lure=lure)
    sym_list  = sp.symbols(list(idx.keys()))    # [x_0, y_0, z_0, phi_0, theta_0, psi_0]
    positions = misc.get_3d_marker_coords({'x': sym_list}, lure=lure, mode=mode)

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
    K_arr, D_arr, R_arr, t_arr, cam_res, cam_names, n_cams = camera_params
    D_arr = D_arr.reshape((-1,4))

    # ========= IMPORT DATA ========
    markers = misc.get_markers(mode=mode, lure=lure)
    Q = np.array([params['Q'][str(s)] for s in sym_list], dtype=np.float64)**2
    R = np.array([params['R'][str(s)] for s in markers], dtype=np.float64)

    proj_funcs = [pt3d_to_x2d, pt3d_to_y2d]

    # save parameters
    if 'markers' in params.keys():
        c = max(params['markers'].values()) + 1
        new_markers = dict(zip(markers, range(len(markers))))
        for k, v in new_markers.items():
            params['markers'][k] = v + c
    else:
        params['markers'] = dict(zip(markers, range(len(markers))))

    if 'state_indices' in params.keys():
        c = max(params['state_indices'].values())
        for k, v in idx.items():
            params['state_indices'][k] = v + c
    else:
        params['state_indices'] = copy.deepcopy(idx)

    if 'skeletons' in params.keys():
        params['skeletons'] += misc.get_skeleton(mode)
    else:
        params['skeletons'] = misc.get_skeleton(mode)

    #===================================================
    #                   Load in data
    #===================================================
    print('----- Generating pairwise 3D points -----')
    points_3d_df = utils.get_pairwise_3d_points_from_df(
        points_2d_df.query(f'likelihood > {dlc_thresh}'),
        K_arr, D_arr, R_arr, t_arr,
        triangulate_points_fisheye
    )
    # points_3d_df, _ = app.sba_points_fisheye(scene_fpath, points_2d_df.query(f'likelihood > {dlc_thresh}'))

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

    m.Ts = 1.0 / params['vid_fps']  # timestep
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

    def init_meas_weights(m, n, c, l):
        likelihood = get_likelihood_from_df(n + start_frame, c, l)
        if likelihood > dlc_thresh:
            return 1 / R[l-1]
        else:
            return 0

    def init_model_weights(m, p):
        return 1 / Q[p-1]

    def init_measurements_df(m, n, c, l, d2):
        return get_meas_from_df(n+start_frame, c, l, d2)

    m.meas_err_weight  = pyo.Param(m.N, m.C, m.L, initialize=init_meas_weights, mutable=True)
    m.model_err_weight = pyo.Param(m.P, initialize=init_model_weights)
    m.meas = pyo.Param(m.N, m.C, m.L, m.D2, initialize=init_measurements_df)

    # ===== MODEL VARIABLES =====
    m.x           = pyo.Var(m.N, m.P) # position
    m.dx          = pyo.Var(m.N, m.P) # velocity
    m.ddx         = pyo.Var(m.N, m.P) # acceleration
    m.poses       = pyo.Var(m.N, m.L, m.D3)
    m.slack_model = pyo.Var(m.N, m.P)
    m.slack_meas  = pyo.Var(m.N, m.C, m.L, m.D2, initialize=0.0)
    if sd:
        if sd_mode == 'const':
            m.shutter_delay = pyo.Var(m.C, initialize=0.0)
        elif sd_mode == 'variable':
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

    print('- init_x:', init_x.shape)
    print('- x_est:', x_est.shape)
    print('- start_frame:', start_frame)
    print('- end_frame:', end_frame)
    x0 = 'x_l' if lure else 'x_0'
    y0 = 'y_l' if lure else 'y_0'
    z0 = 'z_l' if lure else 'z_0'
    init_x[:, idx[x0]]   = x_est[start_frame:end_frame+1]
    init_x[:, idx[y0]]   = y_est[start_frame:end_frame+1]
    init_x[:, idx[z0]]   = z_est[start_frame:end_frame+1]
    if not lure:
        init_x[:, idx['psi_0']] = psi_est   # psi = yaw

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
    pyoidx = {}
    for state in idx:
        pyoidx[state] = idx[state] + 1

    #===== SHUTTER DELAY CONSTRAINTS =====
    print('- Shutter delay')

    def shutter_base_constraint(m, n):
        if sd_mode == 'const':
            return m.shutter_delay[1] == 0.0
        elif sd_mode == 'variable':
            return m.shutter_delay[n, 1] == 0.0

    def shutter_delay_constraint(m, n, c):
        if sd_mode == 'const':
            return (-m.Ts*5, m.shutter_delay[c], m.Ts*5)
        if sd_mode == 'variable':
            return (-m.Ts*5, m.shutter_delay[n, c], m.Ts*5)

    if sd:
        m.shutter_base_constraint = pyo.Constraint(m.N, rule=shutter_base_constraint)
        m.shutter_delay_constraint = pyo.Constraint(m.N, m.C, rule=shutter_delay_constraint)

    #===== POSE CONSTRAINTS =====
    print('- Pose')

    def pose_constraint(m, n, l, d3):
        var_list = [m.x[n,p] for p in m.P]
        [pos] = pos_funcs[l-1](*var_list)   # get 3d points
        return pos[d3-1] == m.poses[n,l,d3]

    m.pose_constraint = pyo.Constraint(m.N, m.L, m.D3, rule=pose_constraint)

    if 'phi_0' in markers:
        m.head_phi_0 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi/6, m.x[n,pyoidx["phi_0"]], np.pi/6))
    if 'theta_0' in markers:
        m.head_theta_0 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi/6, m.x[n,pyoidx["theta_0"]], np.pi/6))
    if 'l_1' in markers:
        m.neck_length = pyo.Constraint(m.N, rule=lambda m,n: (0.2, m.x[n,pyoidx['l_1']], 0.3))
    if 'phi_1' in markers:
        m.neck_phi_1 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi/2, m.x[n,pyoidx["phi_1"]], np.pi/2))
        # m.neck_phi_1 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi/6, m.x[n,pyoidx["phi_1"]], np.pi/6))
    if 'theta_1' in markers:
        m.neck_theta_1 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi/6, m.x[n,pyoidx["theta_1"]], np.pi/6))
    if 'psi_1' in markers:
        m.neck_psi_1 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi/6, m.x[n,pyoidx["psi_1"]], np.pi/6))
    if 'theta_2' in markers:
        m.front_torso_theta_2 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi/6, m.x[n,pyoidx["theta_2"]], np.pi/6))
    if 'theta_3' in markers:
        m.back_torso_theta_3 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi/6, m.x[n,pyoidx["theta_3"]], np.pi/6))
    if 'phi_3' in markers:
        m.back_torso_phi_3 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi/6, m.x[n,pyoidx["phi_3"]], np.pi/6))
    if 'psi_3' in markers:
        m.back_torso_psi_3 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi/6, m.x[n,pyoidx["psi_3"]], np.pi/6))
    if 'theta_4' in markers:
        m.tail_base_theta_4 = pyo.Constraint(m.N, rule=lambda m,n: (-(2/3) * np.pi, m.x[n,pyoidx["theta_4"]], (2/3) * np.pi))
    if 'psi_4' in markers:
        m.tail_base_psi_4 = pyo.Constraint(m.N, rule=lambda m,n: (-(2/3) * np.pi, m.x[n,pyoidx["psi_4"]], (2/3) * np.pi))
    if 'theta_5' in markers:
        m.tail_mid_theta_5 = pyo.Constraint(m.N, rule=lambda m,n: (-(2/3) * np.pi, m.x[n,pyoidx["theta_5"]], (2/3) * np.pi))
    if 'psi_5' in markers:
        m.tail_mid_psi_5 = pyo.Constraint(m.N, rule=lambda m,n: (-(2/3) * np.pi, m.x[n,pyoidx["psi_5"]], (2/3) * np.pi))
    if 'theta_6' in markers:
        # m.l_shoulder_theta_6 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi/2, m.x[n,pyoidx["theta_6"]], np.pi/2))
        m.l_shoulder_theta_6 = pyo.Constraint(m.N, rule=lambda m,n: (-(3/4) * np.pi, m.x[n,pyoidx["theta_6"]], (3/4) * np.pi))
    if 'theta_7' in markers:
        m.l_front_knee_theta_7 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi, m.x[n,pyoidx["theta_7"]], 0))
    if 'theta_8' in markers:
        # m.r_shoulder_theta_8 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi/2, m.x[n,pyoidx["theta_8"]], np.pi/2))
        m.r_shoulder_theta_8 = pyo.Constraint(m.N, rule=lambda m,n: (-(3/4) * np.pi, m.x[n,pyoidx["theta_8"]], (3/4) * np.pi))
    if 'theta_9' in markers:
        m.r_front_knee_theta_9 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi, m.x[n,pyoidx["theta_9"]], 0))
    if 'theta_10' in markers:
        # m.l_hip_theta_10 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi/2, m.x[n,pyoidx["theta_10"]], np.pi/2))
        m.l_hip_theta_10 = pyo.Constraint(m.N, rule=lambda m,n: (-(3/4) * np.pi, m.x[n,pyoidx["theta_10"]], (3/4) * np.pi))
    if 'theta_11' in markers:
        m.l_back_knee_theta_11 = pyo.Constraint(m.N, rule=lambda m,n: (0, m.x[n,pyoidx["theta_11"]], np.pi))
    if 'theta_12' in markers:
        # m.r_hip_theta_12 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi/2, m.x[n,pyoidx["theta_12"]], np.pi/2))
        m.r_hip_theta_12 = pyo.Constraint(m.N, rule=lambda m,n: (-(3/4) * np.pi, m.x[n,pyoidx["theta_12"]], (3/4) * np.pi))
    if 'theta_13' in markers:
        m.r_back_knee_theta_13 = pyo.Constraint(m.N, rule=lambda m,n: (0, m.x[n,pyoidx["theta_13"]], np.pi))
    if 'theta_14' in markers:
        m.l_front_ankle_theta_14 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi/4, m.x[n,pyoidx["theta_14"]], (3/4) * np.pi))
    if 'theta_15' in markers:
        m.r_front_ankle_theta_15 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi/4, m.x[n,pyoidx["theta_15"]], (3/4) * np.pi))
    if 'theta_16' in markers:
        m.l_back_ankle_theta_16 = pyo.Constraint(m.N, rule=lambda m,n: (-(3/4) * np.pi, m.x[n,pyoidx["theta_16"]], 0))
    if 'theta_17' in markers:
        m.r_back_ankle_theta_17 = pyo.Constraint(m.N, rule=lambda m,n: (-(3/4) * np.pi, m.x[n,pyoidx["theta_17"]], 0))

    # ===== MEASUREMENT CONSTRAINTS =====
    print('- Measurement')

    def measurement_constraints(m, n, c, l, d2):
        # m ... model
        # n ... frame
        # c ... camera
        # l ... DLC label
        # project
        K, D, R, t = K_arr[c-1], D_arr[c-1], R_arr[c-1], t_arr[c-1]
        if intermode=='pos':
            x = m.poses[n,l,1]
            y = m.poses[n,l,2]
            z = m.poses[n,l,3]
        else:
            if sd_mode == 'const':
                tau = m.shutter_delay[c]
            elif sd_mode == 'variable':
                tau = m.shutter_delay[n, c]
            if sd and intermode=='vel':
                x = m.poses[n,l,1] + m.dx[n,pyoidx[x0]]*tau
                y = m.poses[n,l,2] + m.dx[n,pyoidx[y0]]*tau
                z = m.poses[n,l,3] + m.dx[n,pyoidx[z0]]*tau
            elif sd and intermode=='acc':
                x = m.poses[n,l,1] + m.dx[n,pyoidx[x0]]*tau + m.ddx[n,pyoidx[x0]]*tau*abs(tau)
                y = m.poses[n,l,2] + m.dx[n,pyoidx[y0]]*tau + m.ddx[n,pyoidx[y0]]*tau*abs(tau)
                z = m.poses[n,l,3] + m.dx[n,pyoidx[z0]]*tau + m.ddx[n,pyoidx[z0]]*tau*abs(tau)

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
                            params['redesc_a'],
                            params['redesc_b'],
                            params['redesc_c'],
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
    x, dx, ddx = [], [], []
    for n in m.N:
        x.append([m.x[n, p].value for p in m.P])
        dx.append([m.dx[n, p].value for p in m.P])
        ddx.append([m.ddx[n, p].value for p in m.P])
    states = dict(
        x=np.array(x),
        dx=np.array(dx),
        ddx=np.array(ddx),
    )
    if sd:
        if sd_mode == 'const':
            sd_state = [[m.shutter_delay[c].value for n in m.N] for c in m.C]
        elif sd_mode == 'variable':
            sd_state = [[m.shutter_delay[n,c].value for n in m.N] for c in m.C]
        states['shutter_delay'] = np.array(sd_state)

    return states