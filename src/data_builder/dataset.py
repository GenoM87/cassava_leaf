import os
import datetime

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
import albumentations as A

class cassavaTrain(Dataset):

    def __init__(self, df, cfg, transforms=None, preprocessing=None, train=True):

        if train:
            df = df[df['fold']!=cfg.DATASET.VALID_FOLD]
        else:
            df = df[df['fold']==cfg.DATASET.VALID_FOLD]

        self.df = df
        self.cfg = cfg
        self.transforms = transforms

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx:int):
        row = self.df.iloc[idx]
        img_id = row['image_id'].split('.')[0]

        path_img = os.path.join(
            self.cfg.DATA_DIR,
            'train_images',
            row['image_id']
        )

        img = cv2.imread(
            path_img, cv2.IMREAD_COLOR
        )

        label = row['label']

        if self.transforms:
            augmented = self.transforms(image=img)
            img = augmented['image']
        
        return img, label, img_id

class cassavaTest(Dataset):

    def __init__(self, df, cfg, transforms=None, preprocessing=None, train=True):

        self.df = df
        self.cfg = cfg
        self.transforms = transforms

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx:int):
        row = self.df.iloc[idx]
        img_id = row['image_id'].split('.')[0]

        path_img = os.path.join(
            self.cfg.DATA_DIR,
            'test_images',
            row['image_id']
        )

        img = cv2.imread(
            path_img, cv2.IMREAD_COLOR
        )

        label = row['label']

        if self.transforms:
            augmented = self.transforms(image=img)
            img = augmented['image']
        
        return img, label, img_id
