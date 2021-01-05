import timm
import torch.nn as nn

class CustomNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = timm.create_model(cfg.MODEL.NAME, pretrained=cfg.MODEL.PRETRAINING)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, cfg.MODEL.NUM_CLASSES_OUT)

    def forward(self, x):
        x = self.model(x)
        return x