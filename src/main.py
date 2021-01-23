import sys, os, time, logging, datetime
from pathlib import Path

import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger, WandbLogger

from engine.fitter import cassavaModel
from config import _C as cfg
from models.create_model import CustomNet, freeze_bn

from data_builder import build_valid_loader, build_train_loader

#Creo lla directory per l'esperimento
path_exp = os.path.join(
    cfg.PROJECT_DIR, 'experiments', cfg.MODEL.NAME, str(datetime.date.today())
)

Path(path_exp).mkdir(parents=True, exist_ok=True)

#Istanzio il logger
path_logger = os.path.join(
    path_exp, f'train-{datetime.datetime.now()}.log'
)

logging.basicConfig(filename=path_logger, level=logging.DEBUG)
logger = logging.getLogger()

wandb.init(config=cfg)
sweep_cfg = wandb.config

if __name__ == "__main__":
    
    for fld in range(0, cfg.DATASET.N_SPLITS):
    #for fld in [0]:
        cfg.DATASET.VALID_FOLD = fld
        
        model = cassavaModel(
            cfg
        )

        wblogger = WandbLogger(
            name='cassava_test',
            project='cassava_kaggle'
        )

        checkpoint = ModelCheckpoint(
            dirpath = path_exp,
            save_weights_only=True,
            monitor = 'val_acc',
            filename='cassava-{epoch:02d}-{val_acc:.4f}',
            mode='max',
        )

        trainer = pl.Trainer(
            #tpu_cores=8,
            gpus = 1,
            #precision=16,
            accumulate_grad_batches=cfg.SOLVER.ACC_GRADIENT,
            max_epochs=cfg.SOLVER.NUM_EPOCHS,
            logger= wblogger,
            default_root_dir=path_exp,
            callbacks = [checkpoint],
        )

        trainer.fit(model)