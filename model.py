import segmentation_models_pytorch as smp
import torch

def create_unetplusplus_model(encoder_name="resnet34", in_channels=1, classes=1):
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=classes,
    )
    return model

def load_trained_model(model_path, device, encoder_name="resnet34", in_channels=1, classes=1):
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=in_channels,
        classes=classes,
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.to(device)
    model.eval()
    return model
