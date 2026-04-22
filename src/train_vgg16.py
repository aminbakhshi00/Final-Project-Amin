# Run directly from main directory: python3 src/train_vgg16.py
import torch.nn as nn
from torchvision import models

from helper import run_training_pipeline


class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        vgg_pretrained = models.vgg16_bn(weights=None)
        self.encoder = nn.Sequential(*list(vgg_pretrained.features.children())[:33])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 19, kernel_size=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


run_training_pipeline(VGG16(), model_name="vgg16")
