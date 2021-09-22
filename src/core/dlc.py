import os
import json
import pandas as pd
from typing import Dict, List
from glob import glob

from lib import app


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