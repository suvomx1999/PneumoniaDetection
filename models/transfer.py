import torch
import torch.nn as nn
from torchvision import models

class TransferModel(nn.Module):
    def __init__(self, model_name='densenet121', freeze_backbone=False):
        super(TransferModel, self).__init__()
        
        if model_name == 'densenet121':
            # Use weights='DEFAULT' which is equivalent to ImageNet weights
            self.model = models.densenet121(weights='DEFAULT')
            if freeze_backbone:
                for param in self.model.features.parameters():
                    param.requires_grad = False
            
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(num_ftrs, 1)
            )
            
        elif model_name == 'resnet18':
            self.model = models.resnet18(weights='DEFAULT')
            if freeze_backbone:
                for param in self.model.parameters():
                    param.requires_grad = False
            
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 1)
            )
        else:
            raise ValueError(f"Model {model_name} not supported. Choose 'densenet121' or 'resnet18'.")

    def forward(self, x):
        return self.model(x)
