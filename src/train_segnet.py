# Run directly from main directory: python3 src/train_segnet.py
# ==================================================================
import torch
import torch.nn as nn

from cityscapes_data import IGNORE_INDEX
from train_helpers import run_training_pipeline


DATA_ROOT = "Data"
SPLIT = "train"
NUM_CLASSES = 20
BOTTLENECK_DILATION_RATE = 2

BATCH_SIZE = 50
EPOCHS = 15
LEARNING_RATE = 1e-4
NUM_WORKERS = 4

SKIP_TRAIN = False
MODEL_SAVE_PATH = "segnet.pt"

OUTPUT_DIR = "outputs/segnet"
VISUALIZE_SAMPLES = 3
EVAL_SAMPLES = 100
SAMPLE_PREFIX = "segnet"


class SegNet(nn.Module):
    def __init__(self, num_classes, bottleneck_dilation_rate=2):
        super(SegNet, self).__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.middle = nn.Sequential(
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=bottleneck_dilation_rate,
                dilation=bottleneck_dilation_rate,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=bottleneck_dilation_rate,
                dilation=bottleneck_dilation_rate,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(32 + 32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool1(x1)

        x3 = self.enc2(x2)
        x4 = self.pool2(x3)

        x5 = self.middle(x4)

        x6 = self.up1(x5)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.dec1(x6)

        x7 = self.up2(x6)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.dec2(x7)

        return self.final(x7)


run_training_pipeline(
    model=SegNet(NUM_CLASSES,
        bottleneck_dilation_rate=BOTTLENECK_DILATION_RATE,
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
