# Run directly from main directory: python3 src/train_vgg16.py
# ==================================================================
import torch.nn as nn
from torchvision import models

from cityscapes_data import IGNORE_INDEX
from train_helpers import run_training_pipeline


DATA_ROOT = "Data"
SPLIT = "train"
NUM_CLASSES = 20
USE_PRETRAINED = True

BATCH_SIZE = 40
EPOCHS = 15
LEARNING_RATE = 1e-4
NUM_WORKERS = 4

SKIP_TRAIN = False
MODEL_SAVE_PATH = "vgg16.pt"

OUTPUT_DIR = "outputs/vgg16"
VISUALIZE_SAMPLES = 3
EVAL_SAMPLES = 100
SAMPLE_PREFIX = "vgg16"


class VGG16(nn.Module):
    def __init__(self, num_classes, use_pretrained=False):
        super(VGG16, self).__init__()

        weights = models.VGG16_BN_Weights.DEFAULT if use_pretrained else None
        vgg_pretrained = models.vgg16_bn(weights=weights)

        features = list(vgg_pretrained.features.children())
        self.encoder = nn.Sequential(*features[:33])

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


run_training_pipeline(
    model=VGG16(NUM_CLASSES, use_pretrained=USE_PRETRAINED),
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
