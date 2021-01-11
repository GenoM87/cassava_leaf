import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform(cfg):

    return A.Compose([
        A.Resize(height=cfg.DATASET.IMG_HEIGHT, width=cfg.DATASET.IMG_WIDTH),     
        A.RandomResizedCrop(cfg.DATASET.IMG_HEIGHT, cfg.DATASET.IMG_HEIGHT, p=cfg.DATASET.P_RANDOMRESCROP),
        A.CoarseDropout(p=cfg.DATASET.P_COARSEDROP),
        A.Cutout(num_holes=cfg.DATASET.NUM_HOLES, p=cfg.DATASET.P_CUTOUT),
        A.HorizontalFlip(cfg.DATASET.P_HORIZONATL_FLIP),
        A.VerticalFlip(cfg.DATASET.P_VERTICAL_FLIP),
        A.HueSaturationValue(
            hue_shift_limit=0.2, 
            sat_shift_limit=0.2, 
            val_shift_limit=0.2, 
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1,0.1), 
            contrast_limit=(-0.1, 0.1), 
            p=0.5
        ),
        A.Transpose(cfg.DATASET.P_TRASPOSE),
        A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=cfg.DATASET.P_SHIFT_SCALE, 
            border_mode=cv2.BORDER_REFLECT
        ),
        A.Normalize(
            mean=[0.42984136, 0.49624753, 0.3129598], 
            std=[0.23297946,0.2358761,0.22365381],
            always_apply=True
        ),
        ToTensorV2()
    ])

def get_valid_transform(cfg):
    return A.Compose([
        A.Resize(height=cfg.DATASET.IMG_HEIGHT, width=cfg.DATASET.IMG_WIDTH),
        A.CenterCrop(height=cfg.DATASET.IMG_HEIGHT, width=cfg.DATASET.IMG_WIDTH, p=1.),
        A.Normalize(
            mean=[0.42984136, 0.49624753, 0.3129598], 
            std=[0.23297946,0.2358761,0.22365381],
            always_apply=True
        ),
        ToTensorV2()
    ])

def get_test_transform(cfg):
    return A.Compose([
        A.Resize(height=cfg.DATASET.IMG_HEIGHT, width=cfg.DATASET.IMG_WIDTH),
        A.Normalize(
            mean=[0.42984136, 0.49624753, 0.3129598], 
            std=[0.23297946,0.2358761,0.22365381],
            always_apply=True
        ),
        ToTensorV2()
    ])