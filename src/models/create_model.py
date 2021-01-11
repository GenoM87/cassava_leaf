import timm
import torch.nn as nn

class CustomNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = timm.create_model(cfg.MODEL.NAME, pretrained=cfg.MODEL.PRETRAINING)

        if 'efficientnet' in cfg.MODEL.NAME:
            n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(n_features, cfg.MODEL.NUM_CLASSES_OUT)
        else:
            n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features, cfg.MODEL.NUM_CLASSES_OUT)

    def forward(self, x):
        x = self.model(x)
        return x

def freeze_bn(model):
    
    for module in model.modules():
        # print(module)
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

    return model