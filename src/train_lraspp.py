# Run directly from main directory: python3 src/train_lraspp.py
import torch.nn as nn
from torchvision import models

from helper import run_training_pipeline


class LRASPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.segmentation.lraspp_mobilenet_v3_large(
            weights=None,
            weights_backbone=None,
            num_classes=19,
        )

    def forward(self, x):
        return self.model(x)["out"]


run_training_pipeline(LRASPP(), model_name="lraspp",batch_size=5)
