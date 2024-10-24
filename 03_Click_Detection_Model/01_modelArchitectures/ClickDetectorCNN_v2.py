# CNN model for click detection channels: 128, 128

import torch
from torch import nn

class ClickDetectorCNN(nn.Module):
    def __init__(self, input_channels, output_shape):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, 
                      out_channels=128, 
                      kernel_size=3,
                      stride=1,
                      padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, 
                      out_channels=128, 
                      kernel_size=3,
                      stride=1,
                      padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128 * 8 * 32,
                      out_features=output_shape),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x