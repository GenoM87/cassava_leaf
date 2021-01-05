import os, sys, time, warnings, datetime, gc
from collections import OrderedDict

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import cv2

import torch
import torch.nn.functional as F
import albumentations as A
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy

from .average import AverageMeter
from models.optimizer import make_optimizer
from models.scheduler import make_scheduler

#TODO: provare ad usare questo
from models.loss import BiTemperedLogisticLoss

class Fitter:
    def __init__(self, model, cfg, train_loader, val_loader, logger, exp_path):
        
        self.experiment_path =  exp_path
        os.makedirs(self.experiment_path, exist_ok=True)

        self.model = model.to(cfg.DEVICE)
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger

        #LOSS FN
        self.criterion = BiTemperedLogisticLoss(
            t1=self.cfg.SOLVER.BIT_T1,
            t2=self.cfg.SOLVER.BIT_T2,
            smoothing=self.cfg.SOLVER.SMOOTHING_LOSS
        )#LabelSmoothingCrossEntropy()

        self.optimizer = make_optimizer(
            self.model, self.cfg
        )

        self.scheduler = make_scheduler(
            self.optimizer, 
            self.cfg,
            self.train_loader
        )

        self.epoch = 0
        self.val_score = 0

        self.logger.info(f'Avvio training {datetime.datetime.now()} con i seguenti parametri:')
        self.logger.info(self.cfg)

    def train(self):
        #Start training loop
        for epoch in range(self.epoch, self.cfg.SOLVER.NUM_EPOCHS):

            if epoch < self.cfg.SOLVER.WARMUP_EPOCHS:
                #Create increasing lr
                lr = np.linspace(
                    start=self.cfg.SOLVER.MIN_LR, 
                    stop=self.cfg.SOLVER.LR, 
                    num=self.cfg.SOLVER.WARMUP_EPOCHS
                )

                for g in self.optimizer.param_groups:
                    g['lr'] = lr[epoch]
                
                self.logger.info(f'[TRAIN]WARMUP: Increasing learning rate to {lr[epoch]}')

            t = time.time()
            summary_loss = self.train_one_epoch()
            self.logger.info(
                f'''[RESULT]: Train. Epoch: {self.epoch},
                summary_loss: {summary_loss.avg:.5f}, 
                time: {(time.time() - t):.3f}'''
            )

            valid_loss, valid_auc, valid_acc = self.validate()

            if self.cfg.SOLVER.SCHEDULER == 'ReduceLROnPlateau':
                self.scheduler.step(valid_loss)
            else:
                self.scheduler.step()

            self.logger.info(
                f'''[RESULT]: Val. Epoch: {self.epoch},
                validation_loss: {valid_loss.avg:.5f},
                Auc Score: {valid_auc:.5f}, 
                Accuracy score: {valid_acc:.5f},
                time: {(time.time() - t):.3f}'''
            )
            self.epoch += 1
            if valid_acc > self.val_score:
                self.model.eval()
                self.save(
                    os.path.join(self.experiment_path, f'{self.epoch}{self.cfg.MODEL.NAME}_best.ckpt'))
                self.val_score = valid_acc

    def train_one_epoch(self):
        self.model.train()
        summary_loss = AverageMeter()
        
        t = time.time()

        train_loader = tqdm(self.train_loader, total=len(self.train_loader), desc='Training')

        for step, (imgs, labels, idxs) in enumerate(train_loader):
            
            self.optimizer.zero_grad()
            batch_size = imgs.shape[0]

            imgs = imgs.to(self.cfg.DEVICE)
            targets = labels.to(self.cfg.DEVICE)

            logits = self.model(imgs)
            
            loss = self.criterion(logits, targets)

            loss.backward()
            self.optimizer.step()

            summary_loss.update(loss.detach().cpu().item(), batch_size)

            train_loader.set_description(
                f'Train Step {step}/{len(self.train_loader)}, ' + \
                f'Learning rate {self.optimizer.param_groups[0]["lr"]}, ' + \
                f'summary_loss: {summary_loss.avg:.5f}, ' + \
                f'time: {(time.time() - t):.3f}'
            )

        return summary_loss

    def validate(self):
        self.model.eval()
        t = time.time()
        summary_loss = AverageMeter()

        val_loader = tqdm(self.val_loader, total=len(self.val_loader), desc='Valid')

        y_true = []
        softmax_preds = []
        preds = []
        for step, (imgs, labels, idxs) in enumerate(val_loader):

            targets = labels.to(self.cfg.DEVICE)
            imgs = imgs.to(self.cfg.DEVICE)
            batch_size = imgs.shape[0]

            with torch.no_grad():
                logits = self.model(imgs)

                loss = self.criterion(logits, targets)

            summary_loss.update(loss, batch_size)

            softmax = torch.nn.Softmax(dim=1)(logits).detach().cpu().numpy()
            prediction_class = np.argmax(a=softmax, axis=1)

            val_loader.set_description(
                f'Valid Step {step}/{len(self.val_loader)}, ' + \
                f'Learning rate {self.optimizer.param_groups[0]["lr"]}, ' + \
                f'summary_loss: {summary_loss.avg:.5f}, ' + \
                f'time: {(time.time() - t):.3f}'
            )

            y_true.append(labels.detach().cpu().numpy())
            softmax_preds.append(softmax)
            preds.append(prediction_class)

        y_true = np.concatenate(y_true, axis=0)
        softmax_preds = np.concatenate(softmax_preds, axis=0)
        preds = np.concatenate(preds, axis=0)

        val_auc = roc_auc_score(
            y_true=y_true, 
            y_score=softmax_preds,
            multi_class='ovr'
        )

        val_acc = accuracy_score(
            y_true=y_true,
            y_pred=preds
        )
        self.logger.info(f'''
            [VALID]Epoch {self.epoch}: validation AUC {val_auc}, validation ACC {val_acc}
        ''')
        return summary_loss, val_auc, val_acc

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_score': self.val_score,
            'epoch': self.epoch,
        }, path)

    def save_model(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'val_score': self.val_score,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.val_score = checkpoint['val_score']
        self.epoch = checkpoint['epoch'] + 1