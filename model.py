import torch.nn as nn
from torchvision import models



def build_model(model_name = "convnext_tiny",num_classes=2, pretrained=True ):
    if model_name == "convnext_tiny":
        model = build_convnext_tiny()
    elif model_name == "resnet18":
        model = build_resnet18()
    elif model_name == "resnet50":
        model = build_resnet50()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model    
  
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


def build_resnet18(num_classes=2, pretrained=True):
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet18(weights=weights)

    # Replace the final fully connected layer
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model


def build_resnet50(num_classes=2, pretrained=True):
    if pretrained:
        weights = models.ResNet50_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet50(weights=weights)

    # Replace the final fully connected layer
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model


