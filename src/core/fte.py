import os
import sys
import json
import copy
import numpy as np
import sympy as sp
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Union
from glob import glob
from time import time
from pprint import pprint
from tqdm import tqdm
from scipy.interpolate import UnivariateSpline
from pyomo.opt import SolverFactory

from lib import misc, utils, app, metric
from lib.calib import project_points_fisheye, triangulate_points_fisheye

from .metrics import save_error_dists


def pyo_i(i: int) -> int:
    return i + 1


def fte(
    out_dir,
    body_points_2d_df, lure_points_2d_df,
    fte_config,
    camera_params,
    markers,
    skeletons,
    start_frame, end_frame,
    body_start_frame, body_end_frame,
    lure_start_frame, lure_end_frame,
    dlc_thresh,
    scene_fpath,
    base_label_fpaths: List[str],
    dlc_pw_label_fpaths: List[str],
    params: Dict = {},
    enable_ppms: bool = False,
    video: bool = True,
    plot: bool = False
) -> str:
    # parameters
    lure = 'lure' in markers
    body = not(len(markers) == 1 and lure)
    shutter_delay_mode = fte_config['shutter_delay']['mode']
    shutter_delay = shutter_delay_mode != 'off'
    interpolation_mode = fte_config['shutter_delay']['interpolation']

    params['start_frame'] = start_frame
    params['end_frame'] = end_frame
    params['body_start_frame'] = body_start_frame
    params['body_end_frame'] = body_end_frame
    if lure:
        params['lure_start_frame'] = lure_start_frame
        params['lure_end_frame'] = lure_end_frame
    params['redesc_a'] = fte_config['cost_func']['redescending']['a']
    params['redesc_b'] = fte_config['cost_func']['redescending']['b']
    params['redesc_c'] = fte_config['cost_func']['redescending']['c']
    params['R'] = fte_config['R']
    params['R_pw1'] = fte_config['R_pw1']
    params['R_pw2'] = fte_config['R_pw2']
    params['Q'] = fte_config['Q']
    params['dlc_thresh'] = dlc_thresh
    params['scene_fpath'] = scene_fpath
    params['shutter_delay_mode'] = shutter_delay_mode
    params['shutter_delay_interpolation'] = interpolation_mode
    params['skeletons'] = skeletons
    params['R_scale'] = fte_config['R_scale']

    if body:
        if 'lure' in markers:
            markers.remove('lure')

        body_state = _fte(
            out_dir,
            base_label_fpaths, dlc_pw_label_fpaths,
            markers,
            misc.get_pose_params(markers),  # {'x_0': 0, 'y_0': 1, 'z_0': 2, ...}
            body_points_2d_df, camera_params,
            body_start_frame, body_end_frame, dlc_thresh,
            params=params,
            enable_ppms=enable_ppms,
            lure=False,
            shutter_delay=shutter_delay,
            shutter_delay_mode=shutter_delay_mode,
            interpolation_mode=interpolation_mode,
        )
    if lure:
        markers = ['lure']
        lure_state = _fte(
            out_dir,
            base_label_fpaths, dlc_pw_label_fpaths,
            markers,
            misc.get_pose_params(markers),  # {'x_l': 0, 'y_l': 1, 'z_l': 2}
            lure_points_2d_df, camera_params,
            lure_start_frame, lure_end_frame, dlc_thresh,
            params=params,
            enable_ppms=False,
            lure=True,
            shutter_delay=shutter_delay,
            shutter_delay_mode=shutter_delay_mode,
            interpolation_mode=interpolation_mode,
        )
    with open(os.path.join(out_dir, 'reconstruction_params.json'), 'w') as f:
        json.dump(params, f)
    pprint(params)

    # reshape with start and end frame
    if body and lure:
        state = {}
        bs = start_frame - body_start_frame
        be = len(body_state['x']) - (body_end_frame - end_frame)
        ls = start_frame - lure_start_frame
        le = len(lure_state['x']) - (lure_end_frame - end_frame)
        for i in ['x', 'dx', 'ddx']:
            state[i] = np.concatenate((body_state[i][bs:be, :], lure_state[i][ls:le, :]), axis=1)
        state['shutter_delay'] = body_state['shutter_delay'][:, bs:be]
        # state['shutter_delay'] = (body_state['shutter_delay'][:, bs:be] + lure_state['shutter_delay'][:, ls:le]) / 2
        # idx
        state['idx'] = body_state['idx']
        state['idx'].update({k: v+len(body_state['idx']) for k,v in lure_state['idx'].items()})
        # marker
        state['marker'] = body_state['marker'] + lure_state['marker']
    elif body:
        state = body_state
    elif lure:
        state = lure_state
    state['skeletons'] = skeletons

    # ========= SAVE FTE RESULTS ========
    k_arr, d_arr, R_arr, t_arr, cam_res, cam_names, n_cams = camera_params
    intermode = interpolation_mode

    # calculate residual error
    positions_3ds = misc.get_all_marker_coords_from_states(state, n_cams, directions=True, intermode=intermode)
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
    pix_errors = metric.residual_error(body_points_2d_df, points_3d_dfs, markers, camera_params)
    save_error_dists(pix_errors, out_dir)
    state['body_reprj_errors'] = pix_errors
    state['lure_reprj_errors'] = metric.residual_error(lure_points_2d_df, points_3d_dfs, markers, camera_params)

    # save pkl/mat and video files
    out_fpath = app.save_fte(state, out_dir, camera_params, start_frame, directions=True, intermode=intermode, save_videos=video)

    # plot cheetah state
    fig_fpath = os.path.join(out_dir, 'fte.pdf')
    app.plot_cheetah_states(state, out_fpath=fig_fpath)
    if shutter_delay:
        fig_fpath = os.path.join(out_dir, 'shutter_delay.pdf')
        app.plot_shutter_delay(state['shutter_delay'], out_fpath=fig_fpath)

    return out_fpath

def _fte(
    out_dir,
    base_label_fpaths, dlc_pw_label_fpaths,
    markers, idx,
    points_2d_df,
    camera_params,
    start_frame, end_frame, dlc_thresh,
    params: Dict = {},
    enable_ppms: bool = False,
    lure: bool = False,
    shutter_delay: bool = False,
    shutter_delay_mode: str = 'const',
    interpolation_mode: str = 'pos',
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
    os.makedirs(out_dir, exist_ok=True)
    app.start_logging(os.path.join(out_dir, 'fte.log'))

    # symbolic vars
    sym_list  = sp.symbols(list(idx.keys()))    # [x_0, y_0, z_0, phi_0, ...]
    positions = misc.get_3d_marker_coords({'x': sym_list}, idx)

    t0 = time()

    # ========= IMPORT CAMERA & SCENE PARAMS ========
    k_arr, d_arr, R_arr, t_arr, cam_res, cam_names, n_cams = camera_params
    d_arr = d_arr.reshape((-1, 4))

    N = end_frame - start_frame + 1    # number of timesteps in trajectory
    assert N > 0
    fps = params['vid_fps']
    Ts = 1.0 / fps

    # ========= POSE FUNCTIONS ========
    func_map = {'sin': pyo.sin, 'cos': pyo.cos, 'ImmutableDenseMatrix': np.array}
    pose_to_3d = sp.lambdify(sym_list, positions, modules=[func_map])
    pos_funcs = []
    for i in range(positions.shape[0]):
        lamb = sp.lambdify(sym_list, positions[i, :], modules=[func_map])
        pos_funcs.append(lamb)

    # ========= PROJECTION FUNCTIONS ========
    def pt3d_to_2d(x, y, z, K, D, R, t):
        x_2d = x*R[0,0] + y*R[0,1] + z*R[0,2] + t.flatten()[0]
        y_2d = x*R[1,0] + y*R[1,1] + z*R[1,2] + t.flatten()[1]
        z_2d = x*R[2,0] + y*R[2,1] + z*R[2,2] + t.flatten()[2]
        # project onto camera plane
        a    = x_2d / z_2d
        b    = y_2d / z_2d
        # fisheye params
        r    = (a**2 + b**2 +1e-12)**0.5
        th   = pyo.atan(r)
        # distortion
        th_D = th * (1 + D[0]*th**2 + D[1]*th**4 + D[2]*th**6 + D[3]*th**8)
        x_P = a*th_D / (r + 1e-12)
        y_P = b*th_D / (r + 1e-12)
        u = K[0,0]*x_P + K[0,2]
        v = K[1,1]*y_P + K[1,2]
        return u, v

    def pt3d_to_x2d(x, y, z, K, D, R, t):
        u = pt3d_to_2d(x, y, z, K, D, R, t)[0]
        return u

    def pt3d_to_y2d(x, y, z, K, D, R, t):
        v = pt3d_to_2d(x, y, z, K, D, R, t)[1]
        return v

    # ========= IMPORT DATA ========
    proj_funcs = [pt3d_to_x2d, pt3d_to_y2d]
    Q = np.array([params['Q'][str(s)] for s in sym_list], dtype=np.float64)**2
    R_pw = np.array(
        [
            [params['R'][str(s)] for s in markers],
            [params['R_pw1'].get(str(s),1e-6) for s in markers],
            [params['R_pw2'].get(str(s),1e-6) for s in markers],
        ],
        dtype=np.float64
    )
    # Provides some extra uncertainty to the measurements to accomodate for the rigid body body assumption.
    R_pw *= params['R_scale']

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

    #===================================================
    #                   Load in data
    #===================================================
    print('Load H5 2D DLC prediction data')

    points_3d_df = utils.get_pairwise_3d_points_from_df(
        points_2d_df.query(f'likelihood > {dlc_thresh}'),
        k_arr, d_arr, R_arr, t_arr,
        triangulate_points_fisheye
    )

    # estimate initial points
    print('Estimate the initial trajectory')
    # Use the cheetahs spine to estimate the initial trajectory with a 3rd degree spline.
    frame_est = np.arange(end_frame+1)

    base_pts = points_3d_df[points_3d_df['marker']==('lure' if lure else 'nose')][['frame', 'x', 'y', 'z']].values
    base_pts[:, 1] = base_pts[:, 1] - 0.055
    base_pts[:, 3] = base_pts[:, 3] + 0.055
    traj_est_x = UnivariateSpline(base_pts[:, 0], base_pts[:, 1])
    traj_est_y = UnivariateSpline(base_pts[:, 0], base_pts[:, 2])
    traj_est_z = UnivariateSpline(base_pts[:, 0], base_pts[:, 3])
    x_est = np.array(traj_est_x(frame_est))
    y_est = np.array(traj_est_y(frame_est))
    z_est = np.array(traj_est_z(frame_est))

    # Calculate the initial yaw.
    dx_est = np.diff(x_est) / Ts
    dy_est = np.diff(y_est) / Ts
    psi_est = np.arctan2(dy_est, dx_est)
    # Duplicate the last heading estimate as the difference calculation returns N-1.
    psi_est = np.append(psi_est, [psi_est[-1]])

    # Remove datafames from memory to conserve memory usage.
    del points_3d_df

    # Obtain base and pairwise measurments.
    pw_data = {}
    base_data = {}
    cam_idx = 0
    for dlc_path, dlc_pw_path in zip(base_label_fpaths, dlc_pw_label_fpaths):
        # Pairwise correspondence data.
        pw_data[cam_idx] = utils.load_pickle(dlc_pw_path)
        df_temp = pd.read_hdf(dlc_path)
        base_data[cam_idx] = list(df_temp.to_numpy())
        cam_idx += 1

    print(f'Start frame: {start_frame}, End frame: {end_frame}, Frame rate: {fps}')

    #===================================================
    #                   Optimisation
    #===================================================
    print('----- Initialising params & Variables -----')
    m = pyo.ConcreteModel(name='Cheetah from measurements')
    m.Ts = Ts

    # ===== SETS =====
    P  = len(sym_list)  # number of pose parameters
    L  = len(markers)   # number of dlc labels per frame
    C  = n_cams         # number of cameras

    m.N  = pyo.RangeSet(N)
    m.P  = pyo.RangeSet(P)
    m.L  = pyo.RangeSet(L)
    m.C  = pyo.RangeSet(C)
    m.D2 = pyo.RangeSet(2)  # dimensionality of measurements (image points)
    m.D3 = pyo.RangeSet(3)  # dimensionality of measurements (3d points)
    m.W = pyo.RangeSet(3 if enable_ppms else 1)

    index_dict = misc.get_dlc_marker_indices()
    pair_dict = misc.get_pairwise_graph()

    # ======= WEIGHTS =======
    def get_likelihood_from_df(n, c, l):
        n_mask = points_2d_df['frame']  == n-1
        l_mask = points_2d_df['marker'] == markers[l-1]
        c_mask = points_2d_df['camera'] == c-1
        val    = points_2d_df[n_mask & l_mask & c_mask]
        if len(val['likelihood'].values) > 0:
            return val['likelihood'].values[0]
        else:
            return 0.0

    def init_meas_weights(m, n, c, l, w):
        # Determine if the current measurement is the base prediction or a pairwise prediction.
        cam_idx = c - 1
        marker = markers[l - 1]
        values = pw_data[cam_idx][n-1 + start_frame]
        likelihoods = values['pose'][2::3]
        if w < 2:
            likelihood = get_likelihood_from_df(start_frame+n, c, l)
        else:
            base = pair_dict[marker][w-2]
            likelihood = likelihoods[base]

        # Filter measurements based on DLC threshold.
        # This does ensures that badly predicted points are not considered in the objective function.
        return 1 / R_pw[w-1][l-1] if likelihood > dlc_thresh else 0.0

    m.meas_err_weight = pyo.Param(
        m.N, m.C, m.L, m.W,
        initialize=init_meas_weights,
        mutable=True
    )
    m.model_err_weight = pyo.Param(
        m.P,
        initialize=lambda m, p: 1 / Q[p-1] if Q[p-1] != 0.0 else 0.0
    )

    # ===== PARAMETERS =====
    def get_meas_from_df(n, c, l, d):
        n_mask = points_2d_df['frame']  == n-1
        l_mask = points_2d_df['marker'] == markers[l-1]
        c_mask = points_2d_df['camera'] == c-1
        d_idx  = {1:'x', 2:'y'}
        val    = points_2d_df[n_mask & l_mask & c_mask]
        if len(val[d_idx[d]].values) > 0:
            return val[d_idx[d]].values[0]
        else:
            return 0.0

    def init_measurements(m, n, c, l, d2, w):
        # Determine if the current measurement is the base prediction or a pairwise prediction.
        cam_idx = c - 1
        marker = markers[l - 1]
        if w < 2:
            return get_meas_from_df(start_frame+n, c, l, d2)
        else:
            try:
                values = pw_data[cam_idx][(n - 1) + start_frame]
                val = values['pose'][d2 - 1::3]
                base = pair_dict[marker][w - 2]
                val_pw = values['pws'][:, :, :, d2 - 1]
                return val[base] + val_pw[0, base, index_dict[marker]]
            except IndexError:
                return 0.0

    m.meas = pyo.Param(m.N, m.C, m.L, m.D2, m.W, initialize=init_measurements)

    # ===== MODEL VARIABLES =====
    m.x           = pyo.Var(m.N, m.P, domain=pyo.Reals) # position
    m.dx          = pyo.Var(m.N, m.P, domain=pyo.Reals) # velocity
    m.ddx         = pyo.Var(m.N, m.P, domain=pyo.Reals) # acceleration
    m.poses       = pyo.Var(m.N, m.L, m.D3, domain=pyo.Reals)
    m.slack_model = pyo.Var(m.N, m.P, domain=pyo.Reals, initialize=0.0)
    m.slack_meas  = pyo.Var(m.N, m.C, m.L, m.D2, m.W, initialize=0.0, domain=pyo.Reals)
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
    print('Variable initialisation...')

    # estimate initial points
    init_x    = np.zeros((N, P))
    init_dx   = np.zeros((N, P))
    init_ddx  = np.zeros((N, P))
    x0 = 'x_l' if lure else 'x_0'
    y0 = 'y_l' if lure else 'y_0'
    z0 = 'z_l' if lure else 'z_0'
    init_x[:, idx[x0]] = x_est[start_frame:end_frame+1]
    init_x[:, idx[y0]] = y_est[start_frame:end_frame+1]
    init_x[:, idx[z0]] = z_est[start_frame:end_frame+1]
    if not lure:
        init_x[:, idx['psi_0']] = psi_est[start_frame:end_frame+1]  # yaw = psi

    for n in m.N:
        for p in m.P:
            if n <= len(init_x): #init using known values
                m.x[n,p].value = init_x[n-1,p-1]
                m.dx[n,p].value = init_dx[n-1,p-1]
                m.ddx[n,p].value = init_ddx[n-1,p-1]
            else: #init using last known value
                m.x[n,p].value = init_x[-1,p-1]
                m.dx[n,p].value = init_dx[-1,p-1]
                m.ddx[n,p].value = init_ddx[-1,p-1]
        # init pose
        var_list = [m.x[n,p].value for p in m.P]
        for l in m.L:
            [pos] = pos_funcs[l-1](*var_list)
            for d3 in m.D3:
                m.poses[n,l,d3].value = pos[d3-1]

    print('Done!')

    # ===== CONSTRAINTS =====
    print('Constaint initialisation...')

    # NOTE: 1 based indexing for pyomo!!!!...@#^!@#&
    pyoidx = {}
    for state in idx:
        pyoidx[state] = idx[state] + 1

    # 3D POSE
    def pose_constraint(m, n, l, d3):
        var_list = [m.x[n,p] for p in m.P]
        [pos] = pos_funcs[l-1](*var_list)   # get 3d points
        return pos[d3-1] == m.poses[n,l,d3]

    m.pose_constraint = pyo.Constraint(m.N, m.L, m.D3, rule=pose_constraint)

    # INTEGRATION
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

    m.integrate_p  = pyo.Constraint(m.N, m.P, rule=backwards_euler_pos)
    m.integrate_v  = pyo.Constraint(m.N, m.P, rule=backwards_euler_vel)

    # MODEL
    def constant_acc(m, n, p):
        if n > 1:
            return m.ddx[n,p] == m.ddx[n-1,p] + m.slack_model[n,p]
        else:
            return pyo.Constraint.Skip

    m.constant_acc = pyo.Constraint(m.N, m.P, rule=constant_acc)

    # SHUTTER DELAY
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

    # MEASUREMENT
    def measurement_constraints(m, n, c, l, d2, w):
        # m ... model
        # n ... frame
        # c ... camera
        # l ... DLC label
        tau = m.shutter_delay[c] if sd else 0.0
        K, D, R, t = k_arr[c-1], d_arr[c-1], R_arr[c-1], t_arr[c-1]
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

        return proj_funcs[d2-1](x, y, z, K, D, R, t) - m.meas[n, c, l, d2, w] - m.slack_meas[n, c, l, d2, w] == 0.0

    m.measurement = pyo.Constraint(m.N, m.C, m.L, m.D2, m.W, rule=measurement_constraints)

    # POSE CONSTRAINTS
    if 'phi_0' in pyoidx.keys():
        m.head_phi_0 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi/6, m.x[n,pyoidx["phi_0"]], np.pi/6)) # 0.52
    if 'theta_0' in pyoidx.keys():
        m.head_theta_0 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi/6, m.x[n,pyoidx["theta_0"]], np.pi/6)) # 0.52
    if 'l_1' in pyoidx.keys():
        m.neck_length = pyo.Constraint(m.N, rule=lambda m,n: (0.2, m.x[n,pyoidx['l_1']], 0.3))
    if 'phi_1' in pyoidx.keys():
        m.neck_phi_1 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi/2, m.x[n,pyoidx["phi_1"]], np.pi/2)) # 1.57
        # m.neck_phi_1 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi/6, m.x[n,pyoidx["phi_1"]], np.pi/6)) # 0.52
    if 'theta_1' in pyoidx.keys():
        m.neck_theta_1 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi/6, m.x[n,pyoidx["theta_1"]], np.pi/6)) # 0.52
    if 'psi_1' in pyoidx.keys():
        m.neck_psi_1 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi/6, m.x[n,pyoidx["psi_1"]], np.pi/6)) # 0.52
    if 'theta_2' in pyoidx.keys():
        m.front_torso_theta_2 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi/6, m.x[n,pyoidx["theta_2"]], np.pi/6)) # 0.52
    if 'theta_3' in pyoidx.keys():
        m.back_torso_theta_3 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi/6, m.x[n,pyoidx["theta_3"]], np.pi/6))
    if 'phi_3' in pyoidx.keys():
        m.back_torso_phi_3 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi/6, m.x[n,pyoidx["phi_3"]], np.pi/6))
    if 'psi_3' in pyoidx.keys():
        m.back_torso_psi_3 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi/6, m.x[n,pyoidx["psi_3"]], np.pi/6))
    if 'theta_4' in pyoidx.keys():
        m.tail_base_theta_4 = pyo.Constraint(m.N, rule=lambda m,n: (-(2/3) * np.pi, m.x[n,pyoidx["theta_4"]], (2/3) * np.pi))
    if 'psi_4' in pyoidx.keys():
        m.tail_base_psi_4 = pyo.Constraint(m.N, rule=lambda m,n: (-(2/3) * np.pi, m.x[n,pyoidx["psi_4"]], (2/3) * np.pi))
    if 'theta_5' in pyoidx.keys():
        m.tail_mid_theta_5 = pyo.Constraint(m.N, rule=lambda m,n: (-(2/3) * np.pi, m.x[n,pyoidx["theta_5"]], (2/3) * np.pi))
    if 'psi_5' in pyoidx.keys():
        m.tail_mid_psi_5 = pyo.Constraint(m.N, rule=lambda m,n: (-(2/3) * np.pi, m.x[n,pyoidx["psi_5"]], (2/3) * np.pi))
    if 'theta_6' in pyoidx.keys():
        # m.l_shoulder_theta_6 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi/2, m.x[n,pyoidx["theta_6"]], np.pi/2))
        m.l_shoulder_theta_6 = pyo.Constraint(m.N, rule=lambda m,n: (-(3/4) * np.pi, m.x[n,pyoidx["theta_6"]], (3/4) * np.pi))
    if 'theta_7' in pyoidx.keys():
        m.l_front_knee_theta_7 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi, m.x[n,pyoidx["theta_7"]], 0))
    if 'theta_8' in pyoidx.keys():
        # m.r_shoulder_theta_8 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi/2, m.x[n,pyoidx["theta_8"]], np.pi/2))
        m.r_shoulder_theta_8 = pyo.Constraint(m.N, rule=lambda m,n: (-(3/4) * np.pi, m.x[n,pyoidx["theta_8"]], (3/4) * np.pi))
    if 'theta_9' in pyoidx.keys():
        m.r_front_knee_theta_9 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi, m.x[n,pyoidx["theta_9"]], 0))
    if 'theta_10' in pyoidx.keys():
        # m.l_hip_theta_10 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi/2, m.x[n,pyoidx["theta_10"]], np.pi/2))
        m.l_hip_theta_10 = pyo.Constraint(m.N, rule=lambda m,n: (-(3/4) * np.pi, m.x[n,pyoidx["theta_10"]], (3/4) * np.pi))
    if 'theta_11' in pyoidx.keys():
        m.l_back_knee_theta_11 = pyo.Constraint(m.N, rule=lambda m,n: (0, m.x[n,pyoidx["theta_11"]], np.pi))
    if 'theta_12' in pyoidx.keys():
        # m.r_hip_theta_12 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi/2, m.x[n,pyoidx["theta_12"]], np.pi/2))
        m.r_hip_theta_12 = pyo.Constraint(m.N, rule=lambda m,n: (-(3/4) * np.pi, m.x[n,pyoidx["theta_12"]], (3/4) * np.pi))
    if 'theta_13' in pyoidx.keys():
        m.r_back_knee_theta_13 = pyo.Constraint(m.N, rule=lambda m,n: (0, m.x[n,pyoidx["theta_13"]], np.pi))
    if 'theta_14' in pyoidx.keys():
        m.l_front_ankle_theta_14 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi/4, m.x[n,pyoidx["theta_14"]], (3/4) * np.pi))
    if 'theta_15' in pyoidx.keys():
        m.r_front_ankle_theta_15 = pyo.Constraint(m.N, rule=lambda m,n: (-np.pi/4, m.x[n,pyoidx["theta_15"]], (3/4) * np.pi))
    if 'theta_16' in pyoidx.keys():
        m.l_back_ankle_theta_16 = pyo.Constraint(m.N, rule=lambda m,n: (-(3/4) * np.pi, m.x[n,pyoidx["theta_16"]], 0))
    if 'theta_17' in pyoidx.keys():
        m.r_back_ankle_theta_17 = pyo.Constraint(m.N, rule=lambda m,n: (-(3/4) * np.pi, m.x[n,pyoidx["theta_17"]], 0))

    print('Done!')

    # ======= OBJECTIVE FUNCTION =======
    print('Objective initialisation...')

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
                        for w in m.W:
                            slack_meas_err += misc.redescending_loss(
                                m.meas_err_weight[n, c, l, w] * m.slack_meas[n, c, l, d2, w],
                                params['redesc_a'],
                                params['redesc_b'],
                                params['redesc_c'],
                            )
        return 1e-3 * (slack_meas_err + slack_model_err)

    m.obj = pyo.Objective(rule=obj)
    print('Done!')

    # run the solver
    opt = SolverFactory(
        'ipopt',
        executable='/tmp/build/bin/ipopt'
    )

    # solver options
    opt.options['print_level']  = 5
    opt.options['max_iter']     = 10000
    opt.options['max_cpu_time'] = 10000
    opt.options['Tol'] = 1e-1
    opt.options['OF_print_timing_statistics'] = 'yes'
    opt.options['OF_print_frequency_iter']    = 10
    opt.options['OF_hessian_approximation']   = 'limited-memory'
    opt.options['OF_accept_every_trial_step'] = 'yes'
    opt.options['linear_solver'] = 'ma86'
    opt.options['OF_ma86_scaling'] = 'none'

    t1 = time()
    print('Initialization took {0:.2f} seconds\n'.format(t1 - t0))

    print('Optimization ...')
    t0 = time()
    results = opt.solve(m, tee=True)
    t1 = time()
    print('\nOptimization took {0:.2f} seconds\n'.format(t1 - t0))

    app.stop_logging()

    # ========= SAVE FTE RESULTS ========
    x_optimised = _get_vals_v(m.x, [m.N, m.P])
    dx_optimised = _get_vals_v(m.dx, [m.N, m.P])
    ddx_optimised = _get_vals_v(m.ddx, [m.N, m.P])
    positions = [pose_to_3d(*states) for states in x_optimised]
    model_weight = _get_vals_v(m.model_err_weight, [m.P])
    model_err = _get_vals_v(m.slack_model, [m.N, m.P])
    meas_err = _get_vals_v(m.slack_meas, [m.N, m.C, m.L, m.D2, m.W])
    meas_weight = _get_vals_v(m.meas_err_weight, [m.N, m.C, m.L, m.W])
    if sd:
        if sd_mode == 'const':
            sd_state = [[m.shutter_delay[c].value for n in m.N] for c in m.C]
        elif sd_mode == 'variable':
            sd_state = [[m.shutter_delay[n,c].value for n in m.N] for c in m.C]
        shutter_delay = np.array(sd_state)
    else:
        shutter_delay = None

    states = dict(
        x=x_optimised,
        dx=dx_optimised,
        ddx=ddx_optimised,
        marker=markers,
        idx=idx,
        model_err=model_err,
        model_weight=model_weight,
        meas_err=meas_err,
        meas_weight=meas_weight,
        shutter_delay=shutter_delay
    )

    return states


def _get_vals_v(var: Union[pyo.Var, pyo.Param], idxs: list) -> np.ndarray:
    '''
    Verbose version that doesn't try to guess stuff for ya. Usage:
    >>> get_vals(m.q, (m.N, m.DOF))
    '''
    arr = np.array([pyo.value(var[idx]) for idx in var]).astype(float)
    return arr.reshape(*(len(i) for i in idxs))