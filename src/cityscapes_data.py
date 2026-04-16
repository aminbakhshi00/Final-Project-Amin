from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMG_HEIGHT = 256
IMG_WIDTH = 512
NUM_CLASSES = 20
# We use 255 as a true ignore label (outside the model class range 0..19).
IGNORE_INDEX = 255
NEAREST = Image.Resampling.NEAREST


def convert_label(label, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
    label = np.array(label, dtype=np.uint8)
    # Keep only labels in [0, num_classes-1]. Everything else is ignored.
    label[(label >= num_classes) & (label != ignore_index)] = ignore_index
    return Image.fromarray(label)


class CityscapesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root="Data",
        split="train",
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        num_classes=NUM_CLASSES,
        ignore_index=IGNORE_INDEX,
        transform=None,
    ):
        self.transform = transform
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        image_dir = Path(data_root) / "leftImg8bit" / split
        label_dir = Path(data_root) / "gtFine" / split

        self.samples = []
        image_paths = sorted(image_dir.glob("*/*_leftImg8bit.png"))

        for img_path in image_paths:
            stem = img_path.name.replace("_leftImg8bit.png", "")
            label_path = label_dir / img_path.parent.name / f"{stem}_gtFine_labelIds.png"
            if label_path.exists():
                self.samples.append((img_path, label_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.img_width, self.img_height))

        label = Image.open(label_path)
        label = label.resize((self.img_width, self.img_height), NEAREST)
        label = convert_label(
            label,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )

        if self.transform:
            image = self.transform(image)

        label = torch.from_numpy(np.array(label)).long()
        return image, label
