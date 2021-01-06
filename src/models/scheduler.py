import torch
from torch.optim import lr_scheduler

def make_scheduler(optimizer, cfg):

    if cfg.SOLVER.SCHEDULER == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=cfg.SOLVER.SCHEDULER_T0, 
            T_mult=cfg.SOLVER.T_MUL, 
            eta_min=cfg.SOLVER.MIN_LR, 
            last_epoch=-1
        )
        return scheduler

    elif cfg.SOLVER.SCHEDULER == 'ReduceLROnPlateau':
        return lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode=cfg.SOLVER.SCHEDULER_MODE,
            factor=cfg.SOLVER.SCHEDULER_REDFACT,
            patience=cfg.SOLVER.SCHEDULER_PATIENCE,
            min_lr=cfg.SOLVER.MIN_LR
        )
    if cfg.SOLVER.SCHEDULER == 'CosineAnnealingLR':    
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=cfg.SOLVER.T_MAX, 
            eta_min=cfg.SOLVER.MIN_LR, 
            last_epoch=-1
        )
        return scheduler
    
    else:
        print('NOME SCHEDULER NON RICONOSCIUTO!')
