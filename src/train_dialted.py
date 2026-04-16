# Run directly from main directory: python3 src/train_dialted.py
# ==================================================================
import torch.nn as nn

from cityscapes_data import IGNORE_INDEX
from train_helpers import run_training_pipeline


DATA_ROOT = "Data"
SPLIT = "train"
NUM_CLASSES = 20
DILATION_RATE = 4

BATCH_SIZE = 100
EPOCHS = 5
LEARNING_RATE = 1e-3
NUM_WORKERS = 0

SKIP_TRAIN = False
MODEL_SAVE_PATH = "dialted.pt"

OUTPUT_DIR = "outputs/dialted"
VISUALIZE_SAMPLES = 3
EVAL_SAMPLES = 100
SAMPLE_PREFIX = "dialted"


class DilatedCNN(nn.Module):
    def __init__(self, num_classes):
        # Same as SimpleCNN, but with a larger dilation while preserving feature-map size.
        super(DilatedCNN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=DILATION_RATE, dilation=DILATION_RATE),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=DILATION_RATE, dilation=DILATION_RATE),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=DILATION_RATE, dilation=DILATION_RATE),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


run_training_pipeline(
    model=DilatedCNN(NUM_CLASSES),
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
