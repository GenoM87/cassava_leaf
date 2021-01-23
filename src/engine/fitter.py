import os, sys, time, warnings, datetime, gc
from collections import OrderedDict

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import cv2

import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import albumentations as A
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy

from .average import AverageMeter
from models.optimizer import make_optimizer
from models.scheduler import make_scheduler
from models.create_model import CustomNet, freeze_bn
from data_builder.builder import build_train_loader, build_valid_loader

from models.loss import BiTemperedLogisticLoss

#TODO: test di SAM
from sam.sam import SAM
from pytorch_ranger import Ranger

class cassavaModel(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()

        model = CustomNet(
            cfg
        )

        if cfg.SOLVER.FREEZE_BN:
            model = freeze_bn(model)

        self.model = model
        del model

        self.train_accuracy = pl.metrics.Accuracy()
        self.valid_accuracy = pl.metrics.Accuracy()
        self.loss_fn = BiTemperedLogisticLoss(
            t1=cfg.SOLVER.BIT_T1,
            t2=cfg.SOLVER.BIT_T2,
            smoothing=cfg.SOLVER.SMOOTHING_LOSS 
        )
        self.cfg = cfg

    def forward(self, x):
        return self.model(x)
      
    def loss(self, logits, labels):
        return self.loss_fn(logits, labels)

    def training_step(self, batch, batch_idx):
        x, y, idxs = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log(
            'loss', 
            loss
        )
        self.log(
            'train_acc_step', 
            self.train_accuracy(y_hat.argmax(dim=-1), y),
            on_step=True, 
            on_epoch=False
        )
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        x, y, idxs = batch
        y_hat = self.model(x)
        val_loss = self.loss(y_hat, y)
        self.log('val_loss', val_loss)
        self.log(
            'val_acc_step', 
            self.valid_accuracy(y_hat.argmax(dim=-1), y),
            on_step=True, 
            on_epoch=False
        )
        return {'val_loss': val_loss}

    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack([output["loss"] for output in outputs]).mean()
        train_acc_mean = self.train_accuracy.compute()
        self.log_dict(
            {"train_loss": train_loss_mean, 
            "train_acc": train_acc_mean, 
            "step": self.current_epoch}
        )

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([output["val_loss"] for output in outputs]).mean()
        valid_acc_mean = self.valid_accuracy.compute()
        log_dict = {"val_loss": val_loss_mean, "val_acc": valid_acc_mean}
        self.log_dict(log_dict, prog_bar=True)
        self.log_dict({"step": self.current_epoch})

    def configure_optimizers(self):
        optimizer = make_optimizer(
            self.model, 
            self.cfg
        )
        scheduler = make_scheduler(
            optimizer, 
            self.cfg,
        )
        return {
        'optimizer': optimizer,
        'lr_scheduler': scheduler,
        'monitor': 'val_loss'
        }

    def train_dataloader(self):
        loader = build_train_loader(self.cfg)
        return loader

    def val_dataloader(self):
        loader = build_valid_loader(self.cfg)
        return loader