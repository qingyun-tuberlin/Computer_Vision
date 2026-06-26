import torch.nn as nn
from torchvision import models

def build_convnext_tiny(num_classes=2, pretrained=True):
    if pretrained:
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
    else:
        weights = None

    model = models.convnext_tiny(weights=weights)

    model.classifier[2] = nn.Linear(
        model.classifier[2].in_features,
        num_classes
    )

    return model
