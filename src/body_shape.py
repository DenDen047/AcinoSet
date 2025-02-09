import os
import sys
import json
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from glob import glob
from time import time
from pprint import pprint
from tqdm import tqdm
from argparse import ArgumentParser
import pyomo.environ as pyo
from pyomo.core.kernel import conic
from pyomo.opt import SolverFactory

from lib import misc, utils, app, metric
from lib.calib import project_points_fisheye, triangulate_points_fisheye
from lib.misc import get_markers


sns.set_theme()     # apply the default theme
plt.style.use(os.path.join('/configs', 'mplstyle.yaml'))


def get_lengths(
    positions,
    markers,
    verbose: bool = True
) -> Dict:
    ni = markers.index('nose')
    ri = markers.index('r_eye')
    li = markers.index('l_eye')

    n_frame, n_marker, _ = positions.shape  # [frame, marker, xyz]

    # minor functions
    def _get_head(nose, r_eye, l_eye):  # [n_frame, xyz]
        n_frame, _ = nose.shape
        v_eyes = l_eye - r_eye
        v_noses = nose - r_eye

        v_heads = []
        for f in range(n_frame):
            v_eye = v_eyes[f, :]
            v_nose = v_noses[f, :]
            norm_eye = v_eye / np.linalg.norm(v_eye)

            v_head = norm_eye * np.dot(norm_eye, v_nose) + r_eye[f, :]
            v_heads.append(v_head)

        return np.array(v_heads)

    def _get_len(arr1, arr2):
        return np.sqrt(np.sum((arr1 - arr2) ** 2, axis=1))

    def _rm_nan(array1):
        nan_array = np.isnan(array1)
        not_nan_array = ~ nan_array
        array2 = array1[not_nan_array]
        return array2

    def _describe(arr):
        return pd.DataFrame(pd.Series(_rm_nan(arr)).describe()).transpose()

    nose = positions[:, ni, :]
    r_eye = positions[:, ri, :]
    l_eye = positions[:, li, :]
    coe = (r_eye + l_eye) / 2   # center of eyes
    head = _get_head(nose, r_eye, l_eye)

    # get lengths
    results = {
        'coe_reye': _get_len(coe, r_eye),
        'coe_leye': _get_len(coe, l_eye),
        'head_nose': _get_len(head, nose),
    }

    # display the results
    if verbose:
        dfs = []
        for k, v in results.items():
            df = _describe(v)
            df.insert(loc=0, column='name', value=[k])
            dfs.append(df)
        stats = pd.concat(dfs, ignore_index=True)
        print(stats)

    return results


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
    positions = np.full((end_frame - start_frame + 1, len(markers), 3), np.nan) # [frame, marker, xyz]
    for i, marker in enumerate(markers):
        marker_pts = points_3d_df[points_3d_df['marker']==marker][['frame', 'x', 'y', 'z']].values
        for frame, *pt_3d in marker_pts:
            positions[int(frame) - start_frame, i] = pt_3d

    return positions, markers


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

    get_lengths(positions, markers)

    # out_fpath = app.save_sba(positions, OUT_DIR, scene_fpath, markers, start_frame)
    # return out_fpath


def dlc(DATA_DIR, OUT_DIR, dlc_thresh, params: Dict = {}) -> Dict:
    video_fpaths = sorted(glob(os.path.join(DATA_DIR, 'cam[1-9].mp4'))) # original vids should be in the parent dir

    # save parameters
    params['dlc_thresh'] = dlc_thresh
    with open(os.path.join(OUT_DIR, 'video_params.json'), 'w') as f:
        json.dump(params, f)

    app.create_labeled_videos(video_fpaths, out_dir=OUT_DIR, draw_skeleton=True, pcutoff=dlc_thresh, lure=False)

    return params


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


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
    points_2d_df = utils.load_dlc_points_as_df(dlc_points_fpaths, verbose=False)
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

    # get 3d positions
    positions, markers = tri(DATA_DIR, points_2d_df, 0, num_frames - 1, args.dlc_thresh, camera_params, scene_fpath, params=vid_params)
    indices = [markers.index(l) for l in ['nose', 'r_eye', 'l_eye']]

    OUT_DIR = os.path.join(DATA_DIR, 'body_shape')

    # get 2d points
    positions = positions[start_frame:end_frame, :, :]
    n_frame, _, _ = positions.shape
    point2ds = []
    for f in range(n_frame):
        n_pt = positions[f, markers.index('nose'), :]
        r_pt = positions[f, markers.index('r_eye'), :]
        l_pt = positions[f, markers.index('l_eye'), :]

        if np.any(np.isnan(n_pt)) or np.any(np.isnan(r_pt)) or np.any(np.isnan(l_pt)):
            continue

        # get the normal vector of the plane made by the three points
        v_n2l = l_pt - n_pt
        v_n2r = r_pt - n_pt
        normal_vector = np.cross(v_n2l, v_n2r)

        # get the rotation matrix to transform the normal vector into z-axis
        rotmat = rotation_matrix_from_vectors(normal_vector, (0,0,1))
        point2d = np.empty((3, 2))
        point2d[:] = np.nan
        nose_2d = np.array([0,0])
        leye_2d = (rotmat @ v_n2l)[:2]
        reye_2d = (rotmat @ v_n2r)[:2]
        # set the center of mass for the origin
        com = np.array([nose_2d, leye_2d, reye_2d]).mean(axis=0)
        point2d[0, :] = nose_2d - com
        point2d[1, :] = leye_2d - com
        point2d[2, :] = reye_2d - com

        point2ds.append(point2d)

    point2ds = np.array(point2ds, dtype=np.float32)     # [n, markers, xy]

    # plot
    fig, ax = plt.subplots()
    for i in range(point2ds.shape[1]):
        ax.scatter(point2ds[:, i, 0], point2ds[:, i, 1], label=markers[i])
    ax.legend()
    ax.axis('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Overlayed Face Labels')
    out_fpath = os.path.join(OUT_DIR, 'step0_overlay_labels.pdf')
    fig.savefig(out_fpath, transparent=True)
    print(f'Saved {out_fpath}\n')

    # fitting each 2d point sets
    model = pyo.ConcreteModel(name='Facial 2D points')

    model.n_frame = point2ds.shape[0]
    model.F = pyo.RangeSet(point2ds.shape[0])   # number of frames
    model.L = pyo.RangeSet(point2ds.shape[1])   # number of labels(markers)
    model.D2 = pyo.RangeSet(point2ds.shape[2])  # x and y

    def init_measurements(m, f, l, d2):
        return point2ds[f-1, l-1, d2-1]
    model.measures = pyo.Param(model.F, model.L, model.D2, initialize=init_measurements)

    model.rot_angle = pyo.Var(model.F, domain=pyo.Reals, bounds=(-np.pi, np.pi), initialize=0.0)
    model.rot_mat = pyo.Var(model.F, model.D2, model.D2, domain=pyo.Reals, initialize=0.0)
    model.translation = pyo.Var(model.F, model.D2, domain=pyo.Reals, initialize=0.0)

    # model.translation_constraint = pyo.Constraint(
    #     model.F,
    #     rule=lambda m, f: sum(m.translation[f,d]**2 for d in m.D2) <= 100**2
    # )
    model.angle2rotmat1 = pyo.Constraint(
        model.F,
        rule=lambda m, f: pyo.cos(m.rot_angle[f]) == m.rot_mat[f, 1, 1]
    )
    model.angle2rotmat2 = pyo.Constraint(
        model.F,
        rule=lambda m, f: -pyo.sin(m.rot_angle[f]) == m.rot_mat[f, 1, 2]
    )
    model.angle2rotmat3 = pyo.Constraint(
        model.F,
        rule=lambda m, f: pyo.sin(m.rot_angle[f]) == m.rot_mat[f, 2, 1]
    )
    model.angle2rotmat4 = pyo.Constraint(
        model.F,
        rule=lambda m, f: pyo.cos(m.rot_angle[f]) == m.rot_mat[f, 2, 2]
    )

    def obj(m):
        cost = 0
        for l in m.L:
            # calculate the center of mass
            com = [
                sum(m.measures[:, l, d]) / m.n_frame
                for d in m.D2
            ]

            # get the sum of lengths with the center of mass
            for f in m.F:
                pt = [
                    sum(m.measures[f,l,e] * m.rot_mat[f,d,e] for e in m.D2) + m.translation[f,d]
                    for d in m.D2
                ]
                cost += sum((pt[d-1] - com[d-1]) ** 2 for d in m.D2)
        return cost
    model.obj = pyo.Objective(rule=obj)

    # run the solver
    opt = SolverFactory(
        'ipopt',
        executable='/tmp/build/bin/ipopt'
    )
    # solver options
    opt.options['tol'] = 1e-10
    opt.options['print_level']  = 5
    opt.options['max_iter']     = 10000
    opt.options['max_cpu_time'] = 10000
    opt.options['OF_print_timing_statistics'] = 'yes'
    opt.options['OF_print_frequency_iter']    = 10
    opt.options['OF_hessian_approximation']   = 'limited-memory'
    opt.options['linear_solver'] = 'ma86'

    results = opt.solve(model, tee=False)

    # extract the result
    points_dict = {}
    for l in model.L:
        points = []
        for f in model.F:
            theta = model.rot_angle[f].value
            rot_mat = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            t = np.array([model.translation[f,d].value for d in model.D2])
            p = point2ds[f-1, l-1, :]
            points.append(rot_mat @ p + t)
        points_dict[markers[l-1]] = np.array(points)   # [frame, xy]

    # plot
    fig, ax = plt.subplots()
    for k, points in points_dict.items():
        ax.scatter(points[:, 0], points[:, 1], label=k)
    ax.legend()
    ax.axis('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Overlayed Face Labels')
    out_fpath = os.path.join(OUT_DIR, 'step1_overlay_labels.pdf')
    fig.savefig(out_fpath, transparent=True)
    print(f'Saved {out_fpath}\n')

    # fitting T-face to point sets
    model = pyo.ConcreteModel(name='T-shape Face')

    model.n_frame = point2ds.shape[0]
    model.F = pyo.RangeSet(point2ds.shape[0])   # number of frames
    model.L = pyo.RangeSet(point2ds.shape[1])   # number of labels(markers)
    model.D2 = pyo.RangeSet(point2ds.shape[2])  # x and y

    def init_measurements(m, f, l, d2):
        return points_dict[markers[l-1]][f-1, d2-1]
    model.measures = pyo.Param(model.F, model.L, model.D2, initialize=init_measurements)

    model.eye_length = pyo.Var(domain=pyo.PositiveReals, initialize=0.03)
    model.nose_length = pyo.Var(domain=pyo.PositiveReals, initialize=0.07)
    model.rot_angle = pyo.Var(domain=pyo.Reals, bounds=(-np.pi, np.pi), initialize=0.0)
    model.rot_mat = pyo.Var(model.D2, model.D2, domain=pyo.Reals, initialize=0.0)
    model.coe_position = pyo.Var(model.D2, domain=pyo.Reals, initialize=0.0)

    model.angle2rotmat1 = pyo.Constraint(rule=lambda m: pyo.cos(m.rot_angle) == m.rot_mat[1, 1])
    model.angle2rotmat2 = pyo.Constraint(rule=lambda m: -pyo.sin(m.rot_angle) == m.rot_mat[1, 2])
    model.angle2rotmat3 = pyo.Constraint(rule=lambda m: pyo.sin(m.rot_angle) == m.rot_mat[2, 1])
    model.angle2rotmat4 = pyo.Constraint(rule=lambda m: pyo.cos(m.rot_angle) == m.rot_mat[2, 2])

    def obj(m):
        cost = 0
        base = [
            [0, -m.nose_length],    # nose
            [m.eye_length, 0],  # r_eye
            [-m.eye_length, 0], # l_eye
        ]
        for l in m.L:
            for f in m.F:
                pt = [
                    sum(m.rot_mat[i,j] * base[l-1][j-1] for j in m.D2) + m.coe_position[i]
                    for i in m.D2
                ]
                cost += sum((pt[d-1] - m.measures[f,l,d]) ** 2 for d in m.D2)
        return cost
    model.obj = pyo.Objective(rule=obj)

    # run the solver
    opt = SolverFactory(
        'ipopt',
        executable='/tmp/build/bin/ipopt'
    )
    # solver options
    opt.options['tol'] = 1e-10
    opt.options['print_level']  = 5
    opt.options['max_iter']     = 10000
    opt.options['max_cpu_time'] = 10000
    opt.options['OF_print_timing_statistics'] = 'yes'
    opt.options['OF_print_frequency_iter']    = 10
    opt.options['OF_hessian_approximation']   = 'limited-memory'
    opt.options['linear_solver'] = 'ma86'

    results = opt.solve(model, tee=True)

    # extract result
    base = np.array([
        [0, -model.nose_length.value],    # nose
        [model.eye_length.value, 0],  # r_eye
        [-model.eye_length.value, 0], # l_eye
    ])
    theta = model.rot_angle.value
    rot_mat = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    coe = np.array([model.coe_position[d].value for d in model.D2])

    # plot
    fig, ax = plt.subplots()
    for k, points in points_dict.items():
        ax.scatter(points[:, 0], points[:, 1], label=k)
        face_pt = rot_mat @ base[markers.index(k),:] + coe
        ax.plot([coe[0], face_pt[0]], [coe[1], face_pt[1]], marker='o')
    ax.legend()
    ax.axis('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Overlayed Face Labels')
    out_fpath = os.path.join(OUT_DIR, 'step2_overlay_labels.pdf')
    fig.savefig(out_fpath, transparent=True)
    print(f'Saved {out_fpath}\n')

    print('eye_length:', model.eye_length.value)
    print('nose_length:', model.nose_length.value / np.sqrt(2))

