import torch
import torch.nn as nn
import torchvision.models as models

def initialize_model(num_classes, feature_extract=True, use_pretrained=True):
    """
    Initialize the ConvNeXt small model for image classification.
    
    Args:
    - num_classes (int): Number of output classes.
    - feature_extract (bool): If True, only update the reshaped layer params.
    - use_pretrained (bool): If True, use pre-trained weights.

    Returns:
    - model_ft (torch.nn.Module): Initialized ConvNeXt small model.
    """
    model_ft = models.convnext_small(pretrained=use_pretrained)
    num_ftrs = model_ft.classifier[2].in_features
    model_ft.classifier[2] = nn.Linear(num_ftrs, num_classes)
    
    if feature_extract:
        for param in model_ft.parameters():
            param.requires_grad = False
    
    return model_ft
