import os

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, GroupKFold

from config import _C as cfg

if __name__ == "__main__":
    
    skf = StratifiedKFold(
        n_splits=cfg.DATASET.N_SPLITS, 
        shuffle=True, 
        random_state=cfg.RANDOM_STATE
    )
    
    df_train = pd.read_csv(
        os.path.join(cfg.DATA_DIR, 'train.csv')
    )

    df_train['fold'] = 0
    for cnt, (trn_idx, val_idx) in enumerate(skf.split(X=df_train['image_id'], y=df_train['label'])):
        df_train.loc[val_idx, 'fold'] = cnt

    df_train.to_csv(
        os.path.join(cfg.DATA_DIR, 'train_folds.csv')
    )