# Run directly from main directory: python3 src/train_lraspp_dialated.py
import torch.nn as nn
from torchvision import models

from helper import run_training_pipeline


class LRASPPDialated(nn.Module):
    def __init__(self):
        super().__init__()
        dilation = 2
        self.model = models.segmentation.lraspp_mobilenet_v3_large(
            weights=None,
            weights_backbone=None,
            num_classes=19,
        )
        for layer in self.model.backbone.modules():
            if isinstance(layer, nn.Conv2d) and layer.kernel_size != (1, 1):
                layer.dilation = (dilation, dilation)
                layer.padding = tuple(((k - 1) * dilation) // 2 for k in layer.kernel_size)

    def forward(self, x):
        return self.model(x)["out"]


run_training_pipeline(LRASPPDialated(), model_name="lraspp_dialated",batch_size=5)
