import sys
import numpy as np
import sympy as sp
from typing import Dict, List
from scipy.spatial.transform import Rotation


def get_markers(mode: str = 'default', directions: bool = False) -> List[str]:
    if mode == 'default':
        s = [
            'nose', 'r_eye', 'l_eye', 'neck_base',
            'spine', 'tail_base', 'tail1', 'tail2',
            'r_shoulder', 'r_front_knee', 'r_front_ankle', # 'r_front_paw',
            'l_shoulder', 'l_front_knee', 'l_front_ankle', # 'l_front_paw',
            'r_hip', 'r_back_knee', 'r_back_ankle', # 'r_back_paw',
            'l_hip', 'l_back_knee', 'l_back_ankle', # 'l_back_paw',
            'lure'
        ]   # excludes paws & lure for now!
    elif mode == 'head':
        s = [
            'nose', 'r_eye', 'l_eye'
        ]
    elif mode == 'upper_body':
        s = [
            'nose', 'r_eye', 'l_eye', 'neck_base',
            'spine',
            'r_shoulder',
            'l_shoulder',
        ]
    elif mode == 'head_stabilize':
        s = [
            'nose', 'r_eye', 'l_eye', 'neck_base',
            'spine'
        ]
    elif mode == 'all':
        s = [
            'nose', 'r_eye', 'l_eye', 'neck_base',
            'spine', 'tail_base', 'tail1', 'tail2',
            'r_shoulder', 'r_front_knee', 'r_front_ankle', 'r_front_paw',
            'l_shoulder', 'l_front_knee', 'l_front_ankle', 'l_front_paw',
            'r_hip', 'r_back_knee', 'r_back_ankle', 'r_back_paw',
            'l_hip', 'l_back_knee', 'l_back_ankle', 'l_back_paw',
            'lure'
        ]

    if directions:
        s += ['coe', 'gaze_target']

    return s


def get_skeleton():
    return [
        ['nose', 'l_eye'], ['nose', 'r_eye'], ['nose', 'neck_base'], ['l_eye', 'neck_base'], ['r_eye', 'neck_base'],
        ['neck_base', 'spine'], ['spine', 'tail_base'], ['tail_base', 'tail1'], ['tail1', 'tail2'],
        ['neck_base', 'r_shoulder'], ['r_shoulder', 'r_front_knee'], ['r_front_knee', 'r_front_ankle'], # ['r_front_ankle', 'r_front_paw'],
        ['neck_base', 'l_shoulder'], ['l_shoulder', 'l_front_knee'], ['l_front_knee', 'l_front_ankle'], # ['l_front_ankle', 'l_front_paw'],
        ['tail_base', 'r_hip'], ['r_hip', 'r_back_knee'], ['r_back_knee', 'r_back_ankle'], # ['r_back_ankle', 'r_back_paw'],
        ['tail_base', 'l_hip'], ['l_hip', 'l_back_knee'], ['l_back_knee', 'l_back_ankle'], # ['l_back_ankle', 'l_back_paw']
    ]   # exludes paws for now!


def get_pose_params(mode: str = 'default') -> Dict[str, List]:
    if mode == 'default':
        states = [
            'x_0', 'y_0', 'z_0',         # head position in inertial
            'phi_0', 'theta_0', 'psi_0', # head rotation in inertial
            'l_1', 'phi_1', 'theta_1', 'psi_1', # neck
            'theta_2',                   # front torso
            'phi_3', 'theta_3', 'psi_3', # back torso   TODO: is phi_3 needed?
            'theta_4', 'psi_4',          # tail_base
            'theta_5', 'psi_5',          # tail_mid
            'theta_6', 'theta_7',        # l_shoulder, l_front_knee
            'theta_8', 'theta_9',        # r_shoulder, r_front_knee
            'theta_10', 'theta_11',      # l_hip, l_back_knee
            'theta_12', 'theta_13',      # r_hip, r_back_knee
            'x_l', 'y_l', 'z_l'          # lure position in inertial
        ]   # exludes paws & lure for now!
    elif mode == 'head':
        states = [
            'x_0', 'y_0', 'z_0',         # head position in inertial
            'phi_0', 'theta_0', 'psi_0', # head rotation in inertial
        ]
    elif mode == 'upper_body' or mode == 'head_stabilize':
        states = [
            'x_0', 'y_0', 'z_0',         # head position in inertial
            'phi_0', 'theta_0', 'psi_0', # head rotation in inertial
            'l_1', 'phi_1', 'theta_1', 'psi_1', # neck
            'theta_2',                   # front torso
        ]

    return dict(zip(states, range(len(states))))


def get_gaze_target(x, r=3):
    func = sp.Matrix if isinstance(x[0], sp.Expr) else np.array
    idx = get_pose_params()

    p_head = func([x[idx['x_0']], x[idx['y_0']], x[idx['z_0']]])
    RI_0  = rot_z(x[idx['psi_0']]) @ rot_x(x[idx['phi_0']]) @ rot_y(x[idx['theta_0']])
    R0_I  = RI_0.T
    gaze_target = p_head + R0_I @ func([r, 0, 0])

    return gaze_target


def get_gaze_target_from_positions(pos_h, n_pos, r_eye_pos, r=3):
    p_head = pos_h
    p_nose = n_pos
    p_r_eye = r_eye_pos
    v_nose = _norm_vector(p_nose - p_head)
    v_reye = _norm_vector(r_eye_pos - p_head)

    # TODO: Check this formulation again
    rotation = Rotation.from_mrp(np.tan(np.pi/4 / 4) * v_reye)
    v = rotation.apply(v_nose)
    gaze_target = p_head + r * v

    return gaze_target


def _norm_vector(v):
    return v / np.linalg.norm(v)


def get_all_marker_coords_from_states(states, n_cam: int, directions: bool = False, mode: str = 'default', intermode: str = 'pos') -> List:
    shutter_delay = states.get('shutter_delay')

    marker_pos_arr = []
    for i in range(n_cam):
        if shutter_delay is not None:
            taus = shutter_delay[i]
            marker_pos = np.array([
                get_3d_marker_coords({'x': x, 'dx': dx, 'ddx': ddx}, tau, directions=directions, mode=mode, intermode=intermode)
                for x, dx, ddx, tau in zip(states['x'], states['dx'], states['ddx'], taus)
            ])  # (timestep, marker_idx, xyz)
        else:
            marker_pos = np.array([get_3d_marker_coords({'x': x}, directions=directions, mode=mode) for x in states['x']]) # (timestep, marker_idx, xyz)
        marker_pos_arr.append(marker_pos)

    return marker_pos_arr


def get_3d_marker_coords(states: Dict, tau: float = 0.0, directions: bool = False, mode: str = 'default', intermode: str = 'pos'):
    """Returns either a numpy array or a sympy Matrix of the 3D marker coordinates (shape Nx3) for a given state vector x.
    """
    x = states['x']
    dx = states.get('dx', None)
    ddx = states.get('ddx', None)
    idx = get_pose_params(mode)
    func = sp.Matrix if isinstance(x[0], sp.Expr) else np.array

    if dx is None or intermode not in ['vel', 'acc']:
        dx = [0] * len(x)
    if ddx is None or intermode not in ['acc']:
        ddx = [0] * len(x)

    if mode == 'default':
        # rotations
        RI_0  = rot_z(x[idx['psi_0']]) @ rot_x(x[idx['phi_0']]) @ rot_y(x[idx['theta_0']])         # head
        R0_I  = RI_0.T
        RI_1  = rot_z(x[idx['psi_1']]) @ rot_x(x[idx['phi_1']]) @ rot_y(x[idx['theta_1']]) @ RI_0  # neck
        R1_I  = RI_1.T
        RI_2  = rot_y(x[idx['theta_2']]) @ RI_1     # front torso
        R2_I  = RI_2.T
        RI_3  = rot_z(x[idx['psi_3']]) @ rot_x(x[idx['phi_3']]) @ rot_y(x[idx['theta_3']]) @ RI_2  # back torso
        R3_I  = RI_3.T
        RI_4  = rot_z(x[idx['psi_4']]) @ rot_y(x[idx['theta_4']]) @ RI_3    # tail base
        R4_I  = RI_4.T
        RI_5  = rot_z(x[idx['psi_5']]) @ rot_y(x[idx['theta_5']]) @ RI_4    # tail mid
        R5_I  = RI_5.T
        RI_6  = rot_y(x[idx['theta_6']]) @ RI_2     # l_shoulder
        R6_I  = RI_6.T
        RI_7  = rot_y(x[idx['theta_7']]) @ RI_6     # l_front_knee
        R7_I  = RI_7.T
        RI_8  = rot_y(x[idx['theta_8']]) @ RI_2     # r_shoulder
        R8_I  = RI_8.T
        RI_9  = rot_y(x[idx['theta_9']]) @ RI_8     # r_front_knee
        R9_I  = RI_9.T
        RI_10 = rot_y(x[idx['theta_10']]) @ RI_3    # l_hip
        R10_I = RI_10.T
        RI_11 = rot_y(x[idx['theta_11']]) @ RI_10   # l_back_knee
        R11_I = RI_11.T
        RI_12 = rot_y(x[idx['theta_12']]) @ RI_3    # r_hip
        R12_I = RI_12.T
        RI_13 = rot_y(x[idx['theta_13']]) @ RI_12   # r_back_knee
        R13_I = RI_13.T

        # positions
        _x = x[idx['x_0']] + dx[idx['x_0']] * tau + ddx[idx['x_0']] * (tau**2)
        _y = x[idx['y_0']] + dx[idx['y_0']] * tau + ddx[idx['y_0']] * (tau**2)
        _z = x[idx['z_0']] + dx[idx['z_0']] * tau + ddx[idx['z_0']] * (tau**2)
        p_head  = func([_x, _y, _z])

        p_l_eye         = p_head         + R0_I  @ func([0, 0.03, 0])
        p_r_eye         = p_head         + R0_I  @ func([0, -0.03, 0])
        p_nose          = p_head         + R0_I  @ func([0.055, 0, -0.055])

        p_neck_base     = p_head         + R1_I  @ func([x[idx['l_1']], 0, 0])
        p_spine         = p_neck_base    + R2_I  @ func([-0.37, 0, 0])

        p_tail_base     = p_spine        + R3_I  @ func([-0.37, 0, 0])
        p_tail_mid      = p_tail_base    + R4_I  @ func([-0.28, 0, 0])
        p_tail_tip      = p_tail_mid     + R5_I  @ func([-0.36, 0, 0])

        p_l_shoulder    = p_neck_base    + R2_I  @ func([-0.04, 0.08, -0.10])
        p_l_front_knee  = p_l_shoulder   + R6_I  @ func([0, 0, -0.24])
        p_l_front_ankle = p_l_front_knee + R7_I  @ func([0, 0, -0.28])

        p_r_shoulder    = p_neck_base    + R2_I  @ func([-0.04, -0.08, -0.10])
        p_r_front_knee  = p_r_shoulder   + R8_I  @ func([0, 0, -0.24])
        p_r_front_ankle = p_r_front_knee + R9_I  @ func([0, 0, -0.28])

        p_l_hip         = p_tail_base    + R3_I  @ func([0.12, 0.08, -0.06])
        p_l_back_knee   = p_l_hip        + R10_I @ func([0, 0, -0.32])
        p_l_back_ankle  = p_l_back_knee  + R11_I @ func([0, 0, -0.25])

        p_r_hip         = p_tail_base    + R3_I  @ func([0.12, -0.08, -0.06])
        p_r_back_knee   = p_r_hip        + R12_I @ func([0, 0, -0.32])
        p_r_back_ankle  = p_r_back_knee  + R13_I @ func([0, 0, -0.25])

        p_lure = func([x[idx['x_l']], x[idx['y_l']], x[idx['z_l']]])

        result = [
            p_nose.T, p_r_eye.T, p_l_eye.T,
            p_neck_base.T, p_spine.T,
            p_tail_base.T, p_tail_mid.T, p_tail_tip.T,
            p_r_shoulder.T, p_r_front_knee.T, p_r_front_ankle.T,
            p_l_shoulder.T, p_l_front_knee.T, p_l_front_ankle.T,
            p_r_hip.T, p_r_back_knee.T, p_r_back_ankle.T,
            p_l_hip.T, p_l_back_knee.T, p_l_back_ankle.T,
            p_lure.T,
        ]
    elif mode == 'head':
        # rotations
        RI_0 = rot_z(x[idx['psi_0']]) @ rot_x(x[idx['phi_0']]) @ rot_y(x[idx['theta_0']])   # head
        R0_I = RI_0.T

        # positions
        _x = x[idx['x_0']] + dx[idx['x_0']] * tau + ddx[idx['x_0']] * (tau**2)
        _y = x[idx['y_0']] + dx[idx['y_0']] * tau + ddx[idx['y_0']] * (tau**2)
        _z = x[idx['z_0']] + dx[idx['z_0']] * tau + ddx[idx['z_0']] * (tau**2)
        p_head  = func([_x, _y, _z])
        # p_l_eye = p_head + R0_I @ func([0, 0.03, 0])
        # p_r_eye = p_head + R0_I @ func([0, -0.03, 0])
        # p_nose  = p_head + R0_I @ func([0.055, 0, -0.055])
        # p_l_eye = p_head + R0_I @ func([0, 0.034705, 0])
        # p_r_eye = p_head + R0_I @ func([0, -0.034705, 0])
        # p_nose  = p_head + R0_I @ func([0.04862, 0, -0.04862])
        p_l_eye = p_head + R0_I @ func([0, 0.038852231676497324, 0])
        p_r_eye = p_head + R0_I @ func([0, -0.038852231676497324, 0])
        p_nose  = p_head + R0_I @ func([0.0571868749393016, 0, -0.0571868749393016])

        result = [
            p_nose.T, p_r_eye.T, p_l_eye.T,
        ]
    elif mode == 'upper_body':
        # rotations
        RI_0 = rot_z(x[idx['psi_0']]) @ rot_x(x[idx['phi_0']]) @ rot_y(x[idx['theta_0']])  # head
        R0_I = RI_0.T
        RI_1 = rot_z(x[idx['psi_1']]) @ rot_x(x[idx['phi_1']]) @ rot_y(x[idx['theta_1']]) @ RI_0  # neck
        R1_I = RI_1.T
        RI_2 = rot_y(x[idx['theta_2']]) @ RI_1     # front torso
        R2_I = RI_2.T

        # positions
        _x = x[idx['x_0']] + dx[idx['x_0']] * tau + ddx[idx['x_0']] * (tau**2)
        _y = x[idx['y_0']] + dx[idx['y_0']] * tau + ddx[idx['y_0']] * (tau**2)
        _z = x[idx['z_0']] + dx[idx['z_0']] * tau + ddx[idx['z_0']] * (tau**2)
        p_head  = func([_x, _y, _z])
        # p_l_eye         = p_head         + R0_I  @ func([0, 0.03, 0])
        # p_r_eye         = p_head         + R0_I  @ func([0, -0.03, 0])
        # p_nose          = p_head         + R0_I  @ func([0.055, 0, -0.055])
        p_l_eye = p_head + R0_I @ func([0, 0.038852231676497324, 0])
        p_r_eye = p_head + R0_I @ func([0, -0.038852231676497324, 0])
        p_nose  = p_head + R0_I @ func([0.0571868749393016, 0, -0.0571868749393016])

        p_neck_base     = p_head         + R1_I  @ func([x[idx['l_1']], 0, 0])
        p_spine         = p_neck_base    + R2_I  @ func([-0.37, 0, 0])

        p_l_shoulder    = p_neck_base    + R2_I  @ func([-0.04, 0.08, -0.10])
        p_r_shoulder    = p_neck_base    + R2_I  @ func([-0.04, -0.08, -0.10])

        result = [
            p_nose.T, p_r_eye.T, p_l_eye.T,
            p_neck_base.T,
            p_spine.T,
            p_r_shoulder.T,
            p_l_shoulder.T,
        ]
    elif mode == 'head_stabilize':
        # rotations
        RI_0 = rot_z(x[idx['psi_0']]) @ rot_x(x[idx['phi_0']]) @ rot_y(x[idx['theta_0']])  # head
        R0_I = RI_0.T
        RI_1 = rot_z(x[idx['psi_1']]) @ rot_x(x[idx['phi_1']]) @ rot_y(x[idx['theta_1']]) @ RI_0  # neck
        R1_I = RI_1.T
        RI_2 = rot_y(x[idx['theta_2']]) @ RI_1     # front torso
        R2_I = RI_2.T

        # positions
        _x = x[idx['x_0']] + dx[idx['x_0']] * tau + ddx[idx['x_0']] * (tau**2)
        _y = x[idx['y_0']] + dx[idx['y_0']] * tau + ddx[idx['y_0']] * (tau**2)
        _z = x[idx['z_0']] + dx[idx['z_0']] * tau + ddx[idx['z_0']] * (tau**2)
        p_head  = func([_x, _y, _z])
        # p_l_eye         = p_head         + R0_I  @ func([0, 0.03, 0])
        # p_r_eye         = p_head         + R0_I  @ func([0, -0.03, 0])
        # p_nose          = p_head         + R0_I  @ func([0.055, 0, -0.055])
        p_l_eye = p_head + R0_I @ func([0, 0.038852231676497324, 0])
        p_r_eye = p_head + R0_I @ func([0, -0.038852231676497324, 0])
        p_nose  = p_head + R0_I @ func([0.0571868749393016, 0, -0.0571868749393016])

        p_neck_base     = p_head         + R1_I  @ func([x[idx['l_1']], 0, 0])
        # p_neck_base     = p_head         + R1_I  @ func([-0.28, 0, 0])
        p_spine         = p_neck_base    + R2_I  @ func([-0.37, 0, 0])

        result = [
            p_nose.T, p_r_eye.T, p_l_eye.T,
            p_neck_base.T,
            p_spine.T,
        ]

    if directions:
        p_gaze_target = p_head + R0_I @ func([3, 0, 0])
        result += [p_head.T, p_gaze_target.T]

    return func(result)


def redescending_loss(err, a, b, c):
    # outlier rejecting cost function
    def func_step(start, x):
        return 1/(1+np.e**(-1*(x - start)))

    def func_piece(start, end, x):
        return func_step(start, x) - func_step(end, x)

    e = abs(err)
    cost = 0.0
    cost += (1 - func_step(a, e))/2*e**2
    cost += func_piece(a, b, e)*(a*e - (a**2)/2)
    cost += func_piece(b, c, e)*(a*b - (a**2)/2 + (a*(c-b)/2)*(1-((c-e)/(c-b))**2))
    cost += func_step(c, e)*(a*b - (a**2)/2 + (a*(c-b)/2))
    return cost


def global_positions(R_arr, t_arr):
    "Returns a vector of camera position vectors in the world frame"
    R_arr = np.array(R_arr).reshape((-1, 3, 3))
    t_arr = np.array(t_arr).reshape((-1, 3, 1))

    positions = []
    assert R_arr.shape[0]==t_arr.shape[0], 'Number of cams in R_arr do not match t_arr'
    for r, t in zip(R_arr, t_arr):
        pos = -r.T @ t
        positions.append(pos)

    return np.array(positions, dtype=np.float32)


def rotation_matrix_from_vectors(u,v):
    """ Find the rotation matrix that aligns u to v
    :param u: A 3D "source" vector
    :param v: A 3D "destination" vector
    :return mat: A transform matrix (3x3) which when applied to u, aligns it with v.
    """
    # https://stackoverflow.com/questions/36409140/create-a-rotation-matrix-from-2-normals
    # Suppose you want to write the rotation that maps a vector u to a vector v.
    # if U and V are their unit vectors then W = U^V (cross product) is the axis of rotation and is an invariant
    # Let M be the associated matrix.
    # We have finally: (V,W,V^W) = M.(U,W,U^W)

    U = (u/np.linalg.norm(u)).reshape(3)
    V = (v/np.linalg.norm(v)).reshape(3)

    W = np.cross(U, V)
    A = np.array([U, W, np.cross(U, W)]).T
    B = np.array([V, W, np.cross(V, W)]).T
    return np.dot(B, np.linalg.inv(A))


def rot_x(x):
    if isinstance(x, sp.Expr):
        c = sp.cos(x)
        s = sp.sin(x)
        func = sp.Matrix
    else:
        c = np.cos(x)
        s = np.sin(x)
        func = np.array
    return func([[1, 0, 0],
                 [0, c, s],
                 [0, -s, c]])


def rot_y(y):
    if isinstance(y, sp.Expr):
        c = sp.cos(y)
        s = sp.sin(y)
        func = sp.Matrix
    else:
        c = np.cos(y)
        s = np.sin(y)
        func = np.array
    return func([[c, 0, -s],
                 [0, 1, 0],
                 [s, 0, c]])


def rot_z(z):
    if isinstance(z, sp.Expr):
        c = sp.cos(z)
        s = sp.sin(z)
        func = sp.Matrix
    else:
        c = np.cos(z)
        s = np.sin(z)
        func = np.array
    return func([[c, s, 0],
                 [-s, c, 0],
                 [0, 0, 1]])


# https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
class Logger:

    def __init__(self, out_fpath):
        self.terminal = sys.stdout
        self.logfile = open(out_fpath, 'w', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
