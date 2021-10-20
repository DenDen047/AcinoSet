import os, sys
import json
import pickle
import pandas as pd
from pprint import pprint
from typing import Dict, List
from glob import glob
from scipy.io import savemat

from lib import app, misc


def detect_start_end_frame(points_2d_df, dlc_thresh: float, mode: str = 'default') -> [int, int]:
    # Automatically set start and end frame
    # defining the first and end frame as detecting all the markers on any of cameras simultaneously
    filtered_points_2d_df = points_2d_df.query(f'likelihood > {dlc_thresh}')    # ignore points with low likelihood
    target_markers = misc.get_markers(mode)

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
    start_frames = []
    end_frames = []
    max_idx = int(filtered_points_2d_df['frame'].max() + 1)
    for marker in target_markers:
        for i in range(max_idx):    # start_frame
            if frame_condition_with_key_markers(i, [marker], 2):
                start_frames.append(i)
                break
        for i in range(max_idx, 0, -1): # end_frame
            if frame_condition_with_key_markers(i, [marker], 2):
                end_frames.append(i)
                break
    if len(start_frames)==0 or len(end_frames)==0:
        raise('Setting frames failed. Please define start and end frames manually.')
    else:
        start_frame = max(start_frames)
        end_frame = min(end_frames)

    return start_frame, end_frame


def dlc(DATA_DIR, OUT_DIR, mode, dlc_thresh, params: Dict = {}, video: bool = False) -> Dict:
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

    # extract likelihoods
    markers = misc.get_markers(mode)
    likelihood_data = {}
    for i, df in enumerate(point2d_dfs):
        df = df.loc[:,('DLC_resnet152_CheetahOct14shuffle1_500000')]
        likelihoods = df.loc[:, (markers, 'likelihood')]
        likelihoods.columns = likelihoods.columns.droplevel('coords')
        r = {name: col.values for name, col in likelihoods.items()}
        likelihood_data[f'cam{i+1}'] = r

    # save
    data_fpath = os.path.join(OUT_DIR, 'dlc')
    pkl_fpath = data_fpath + '.pickle'
    with open(pkl_fpath, 'wb') as f:
        pickle.dump(likelihood_data, f)
    mat_fpath = data_fpath + '.mat'
    savemat(mat_fpath, likelihood_data)

    # video
    if video:
        app.create_labeled_videos(point2d_dfs, video_fpaths, out_dir=OUT_DIR, draw_skeleton=True, pcutoff=dlc_thresh, lure=False)

    return params