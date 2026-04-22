import numpy as np
import torch
from torchvision.datasets import Cityscapes
from torchvision.transforms.functional import to_tensor

DATA_ROOT = "Data"
IGNORE_INDEX = 255

# Cityscapes default mapping: label id -> train id (0..18), everything else -> 255.
ID_TO_TRAIN_ID = torch.full((256,), IGNORE_INDEX, dtype=torch.long)
for city_class in Cityscapes.classes:
    if 0 <= city_class.id < 256 and 0 <= city_class.train_id < 19:
        ID_TO_TRAIN_ID[city_class.id] = city_class.train_id


def cityscapes_transforms(image, target):
    image = to_tensor(image)
    target = torch.from_numpy(np.array(target, dtype=np.uint8)).long()
    target = ID_TO_TRAIN_ID[target]
    return image, target


def get_cityscapes_dataset(data_root=DATA_ROOT, split="train"):
    return Cityscapes(
        root=data_root,
        split=split,
        mode="fine",
        target_type="semantic",
        transforms=cityscapes_transforms,
    )
