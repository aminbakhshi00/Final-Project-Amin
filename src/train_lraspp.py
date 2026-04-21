# Run directly from main directory: python3 src/train_lraspp.py
# ==================================================================
import torch.nn as nn
from torchvision import models

from cityscapes_data import IGNORE_INDEX
from train_helpers import run_training_pipeline


DATA_ROOT = "Data"
SPLIT = "train"
NUM_CLASSES = 20
USE_PRETRAINED = True

BATCH_SIZE = 50
EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_WORKERS = 4

SKIP_TRAIN = False
MODEL_SAVE_PATH = "lraspp.pt"

OUTPUT_DIR = "outputs/lraspp"
VISUALIZE_SAMPLES = 3
EVAL_SAMPLES = 100
SAMPLE_PREFIX = "lraspp"


class LRASPP(nn.Module):
    def __init__(self, num_classes, use_pretrained=False):
        super(LRASPP, self).__init__()

        weights_backbone = (
            models.MobileNet_V3_Large_Weights.DEFAULT if use_pretrained else None
        )
        self.model = models.segmentation.lraspp_mobilenet_v3_large(
            weights=None,
            weights_backbone=weights_backbone,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)["out"]


run_training_pipeline(
    model=LRASPP(NUM_CLASSES, use_pretrained=USE_PRETRAINED),
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
