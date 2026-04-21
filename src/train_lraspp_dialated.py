# Run directly from main directory: python3 src/train_lraspp_dialated.py
# ==================================================================
import torch.nn as nn
from torchvision import models

from cityscapes_data import IGNORE_INDEX
from train_helpers import run_training_pipeline


DATA_ROOT = "Data"
SPLIT = "train"
NUM_CLASSES = 20
USE_PRETRAINED = True
DILATION_RATE = 2

BATCH_SIZE = 50
EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_WORKERS = 4

SKIP_TRAIN = False
MODEL_SAVE_PATH = "lraspp_dialated.pt"

OUTPUT_DIR = "outputs/lraspp_dialated"
VISUALIZE_SAMPLES = 3
EVAL_SAMPLES = 100
SAMPLE_PREFIX = "lraspp_dialated"


class LRASPPDialated(nn.Module):
    def __init__(self, num_classes, use_pretrained=False, dilation_rate=2):
        super(LRASPPDialated, self).__init__()

        weights_backbone = (
            models.MobileNet_V3_Large_Weights.DEFAULT if use_pretrained else None
        )
        self.model = models.segmentation.lraspp_mobilenet_v3_large(
            weights=None,
            weights_backbone=weights_backbone,
            num_classes=num_classes,
        )
        self._apply_backbone_dilation(dilation_rate)

    def _apply_backbone_dilation(self, dilation_rate):
        for layer in self.model.backbone.modules():
            if isinstance(layer, nn.Conv2d) and layer.kernel_size != (1, 1):
                layer.dilation = (dilation_rate, dilation_rate)
                layer.padding = tuple(
                    ((kernel_size - 1) * dilation_rate) // 2
                    for kernel_size in layer.kernel_size
                )

    def forward(self, x):
        return self.model(x)["out"]


run_training_pipeline(
    model=LRASPPDialated(
        NUM_CLASSES,
        use_pretrained=USE_PRETRAINED,
        dilation_rate=DILATION_RATE,
    ),
    data_root=DATA_ROOT,
    split=SPLIT,
    num_classes=NUM_CLASSES,
    ignore_index=IGNORE_INDEX,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    num_workers=NUM_WORKERS,
    skip_train=SKIP_TRAIN,
    model_save_path=MODEL_SAVE_PATH,
    output_dir=OUTPUT_DIR,
    visualize_samples=VISUALIZE_SAMPLES,
    eval_samples=EVAL_SAMPLES,
    sample_prefix=SAMPLE_PREFIX,
)
