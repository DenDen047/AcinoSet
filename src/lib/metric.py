import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Union

from . import calib


def _norm_vector(v):
    return v / np.linalg.norm(v)


def get_key_angles(positions, markers):
    # positions
    p_head = np.squeeze(positions[:, markers.index('coe'), :])
    p_gaze = np.squeeze(positions[:, markers.index('gaze_target'), :])
    p_lure = np.squeeze(positions[:, markers.index('lure'), :])

    # vectors
    v_gaze = p_gaze - p_head
    v_lure = p_lure - p_head
    v_gaze = np.apply_along_axis(_norm_vector, axis=1, arr=v_gaze)
    print(v_lure)
    v_lure = np.apply_along_axis(_norm_vector, axis=1, arr=v_lure)
    # print(v_lure)

    # alpha ... angle difference between v_gaze and v_lure
    alpha = []
    for g, l in zip(v_gaze, v_lure):
        v = np.arccos(np.dot(g, l))
        alpha.append(v)
    print(alpha)
    sys.exit(1)

    return


def residual_error(points_2d_df, points_3d_df, markers, camera_params) -> Dict:
    k_arr, d_arr, r_arr, t_arr, _, _ = camera_params
    n_cam = len(k_arr)
    error = {str(i): None for i in range(n_cam)}
    for i in range(n_cam):
        for m in markers:
            # extract frames
            q = f'marker == "{m}"'
            pts_2d_df = points_2d_df.query(q + f'and camera == {i}')
            pts_3d_df = points_3d_df.query(q)
            pts_2d_df = pts_2d_df[pts_2d_df[['x', 'y']].notnull().all(axis=1)]
            pts_3d_df = pts_3d_df[pts_3d_df[['x', 'y', 'z']].notnull().all(axis=1)]
            valid_frames = np.intersect1d(pts_2d_df['frame'].to_numpy(), pts_3d_df['frame'].to_numpy())
            pts_2d_df = pts_2d_df[pts_2d_df['frame'].isin(valid_frames)].sort_values(by=['frame'])
            pts_3d_df = pts_3d_df[pts_3d_df['frame'].isin(valid_frames)].sort_values(by=['frame'])

            # get 2d and reprojected points
            frames = pts_2d_df.query(q)['frame'].to_numpy()
            pts_2d = pts_2d_df.query(q)[['x', 'y']].to_numpy()
            pts_3d = pts_3d_df.query(q)[['x', 'y', 'z']].to_numpy()
            if len(pts_2d_df) == 0 or len(pts_3d_df) == 0:
                continue
            prj_2d = calib.project_points_fisheye(pts_3d, k_arr[i], d_arr[i], r_arr[i], t_arr[i])

            # camera distance
            cam_pos = np.squeeze(t_arr[i, :, :])
            cam_dist = np.sqrt(np.sum((pts_3d - cam_pos) ** 2, axis=1))

            # compare both types of points
            residual = np.sqrt(np.sum((pts_2d - prj_2d) ** 2, axis=1))
            error_uv = pts_2d - prj_2d

            # make the result dataframe
            df = pd.DataFrame(
                np.vstack((frames, cam_dist, residual, error_uv.T)).T,
                columns=['frame', 'camera_distance', 'pixel_residual', 'error_u', 'error_v']
            )
            error[str(i)] = df

    return error
