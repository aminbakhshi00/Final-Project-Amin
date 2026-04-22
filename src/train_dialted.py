# Run directly from main directory: python3 src/train_dialted.py
import torch.nn as nn

from helper import run_training_pipeline


class DilatedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        dilation = 4
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 19, 1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


run_training_pipeline(DilatedCNN(), model_name="dialted")
