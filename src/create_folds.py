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

    #df_train['image_id'] = 'train_images/' +df_train['image_id']

    #df_train_2019 = pd.read_csv(
    #    os.path.join(cfg.DATA_DIR, 'psl_cassava2019.csv')
    #)

    #df_train_2019['image_id'] = df_train_2019['image_id'].str.split('data/', expand=True)[1]
    #df_train_2019.rename(columns={'pred_c': 'label', 1: 'image_id'}, inplace=True)
    #df_train_2019 = df_train_2019[df_train_2019['pred_p']>cfg.DATASET.PSEUDO_THR]

    #df_train = pd.concat([df_train, df_train_2019[['image_id', 'label']]]).reset_index(drop=True)

    df_train['fold'] = 0
    for cnt, (trn_idx, val_idx) in enumerate(skf.split(X=df_train['image_id'], y=df_train['label'])):
        df_train.loc[val_idx, 'fold'] = cnt

    df_train.to_csv(
        os.path.join(cfg.DATA_DIR, 'train_folds.csv'),
        index=False
    )