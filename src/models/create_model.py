import timm

def build_model(cfg):

    model = timm.create_model(
        model_name=cfg.MODEL.NAME,
        pretrained=cfg.MODEL.PRETRAINING,
        num_classes=cfg.MODEL.NUM_CLASSES_OUT #5 Classi, è hardcoded nel codice
    )

    return model