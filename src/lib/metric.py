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

    return


def residual_error(points_2d_df, points_3d_dfs, markers, camera_params) -> Dict:
    k_arr, d_arr, r_arr, t_arr, _, _ = camera_params
    n_cam = len(k_arr)
    if not isinstance(points_3d_dfs, list):
        points_3d_dfs = [points_3d_dfs] * n_cam
    error = {str(i): None for i in range(n_cam)}
    for i in range(n_cam):
        # Calculate the nose to eye distance.
        nose_df = points_2d_df.query(f'marker == "nose" and camera == {i}').sort_values(by=['frame'])
        l_eye_df = points_2d_df.query(f'marker == "l_eye" and camera == {i}').sort_values(by=['frame'])
        r_eye_df = points_2d_df.query(f'marker == "r_eye" and camera == {i}').sort_values(by=['frame'])
        eye_df = l_eye_df.combine_first(r_eye_df)
        eye_df = eye_df.drop_duplicates(subset=["frame"], keep="first")
        valid_frame_range = np.intersect1d(
            points_2d_df.query(f'camera == {i}')['frame'].to_numpy(), points_3d_dfs[i]['frame'].to_numpy())
        valid_frames = np.intersect1d(eye_df['frame'].to_numpy(), nose_df['frame'].to_numpy())
        valid_frames = np.intersect1d(valid_frames, valid_frame_range)
        nose_df = nose_df[nose_df['frame'].isin(valid_frames)]
        eye_df = eye_df[eye_df['frame'].isin(valid_frames)]
        frames = nose_df['frame'].to_numpy()
        nose_pts = nose_df[['x', 'y']].to_numpy()
        eye_pts = eye_df[['x', 'y']].to_numpy()
        nose_to_eye = np.linalg.norm(nose_pts - eye_pts, axis=1)
        nose_to_eye_df = pd.DataFrame(np.vstack((frames, nose_to_eye)).T, columns=['frame', 'distance'])
        nose_to_eye_df = nose_to_eye_df.set_index("frame")
        error_dfs = []
        for m in markers:
            # extract frames
            q = f'marker == "{m}"'
            pts_2d_df = points_2d_df.query(q + f'and camera == {i}')
            pts_3d_df = points_3d_dfs[i].query(q)
            pts_2d_df = pts_2d_df[pts_2d_df[['x', 'y']].notnull().all(axis=1)]
            pts_3d_df = pts_3d_df[pts_3d_df[['x', 'y', 'z']].notnull().all(axis=1)]
            valid_frames = np.intersect1d(pts_2d_df['frame'].to_numpy(), pts_3d_df['frame'].to_numpy())
            pts_2d_df = pts_2d_df[pts_2d_df['frame'].isin(valid_frames)].sort_values(by=['frame'])
            pts_3d_df = pts_3d_df[pts_3d_df['frame'].isin(valid_frames)].sort_values(by=['frame'])
            # pck_distance = nose_to_eye_df[nose_to_eye_df['frame'].isin(valid_frames)].sort_values(by=['frame'])
            pck_distance = nose_to_eye_df.reindex(valid_frames)
            # get 2d and reprojected points
            frames = pts_2d_df['frame'].to_numpy()
            pts_2d = pts_2d_df[['x', 'y']].to_numpy()
            pts_3d = pts_3d_df[['x', 'y', 'z']].to_numpy()
            distance_threshold = pck_distance["distance"].to_numpy()
            if len(pts_2d) == 0 or len(pts_3d) == 0:
                continue
            prj_2d = calib.project_points_fisheye(pts_3d, k_arr[i], d_arr[i], r_arr[i], t_arr[i])
            # camera distance
            cam_pos = np.squeeze(t_arr[i, :, :])
            cam_dist = np.sqrt(np.sum((pts_3d - cam_pos)**2, axis=1))
            # compare both types of points
            residual = np.sqrt(np.sum((pts_2d - prj_2d)**2, axis=1))
            error_uv = pts_2d - prj_2d
            # make the result dataframe
            marker_arr = np.array([m] * len(frames))
            error_dfs.append(
                pd.DataFrame(
                    np.vstack((frames, marker_arr, cam_dist, residual, distance_threshold, error_uv.T)).T,
                    columns=[
                        'frame', 'marker', 'camera_distance', 'pixel_residual', 'pck_threshold', 'error_u', 'error_v'
                    ]
                )
            )
        error[str(i)] = pd.concat(error_dfs, ignore_index=True) if len(error_dfs) > 0 else pd.DataFrame(
            columns=['frame', 'marker', 'camera_distance', 'pixel_residual', 'pck_threshold', 'error_u', 'error_v']
        )
    return error
