import os, sys
import json
import pickle
import pandas as pd
from pprint import pprint
from typing import Dict, List
from glob import glob
from scipy.io import savemat

from lib import app, misc


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