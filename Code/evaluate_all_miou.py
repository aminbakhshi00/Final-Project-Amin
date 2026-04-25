# Run from main directory: python3 src/evaluate_all_miou.py
import torch
import torch.nn as nn
from torchvision import models

from dataloader import get_cityscapes_dataset
from helper import DEVICE, evaluate_miou, load_model_checkpoint

EVAL_SAMPLES = 100


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 19, 1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class DilatedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        dilation = 4
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 19, 1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


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


class VGG16Dialated(nn.Module):
    def __init__(self):
        super().__init__()
        vgg_pretrained = models.vgg16_bn(weights=None)
        features = list(vgg_pretrained.features.children())
        for layer in features:
            if isinstance(layer, nn.Conv2d) and layer.kernel_size == (3, 3):
                layer.dilation = (2, 2)
                layer.padding = (2, 2)
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
            nn.Conv2d(64, 19, kernel_size=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class SegNet(nn.Module):
    def __init__(self):
        super().__init__()
        dilation = 2
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
            nn.Conv2d(64, 128, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.final = nn.Conv2d(32, 19, kernel_size=1)

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


class SegNetDialated(nn.Module):
    def __init__(self):
        super().__init__()
        dilation = 4
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2)
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.final = nn.Conv2d(32, 19, kernel_size=1)

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


def build_models_to_check():
    return [
        ("simple.pt", SimpleCNN()),
        ("dialted.pt", DilatedCNN()),
        ("vgg16.pt", VGG16()),
        ("vgg16_dialated.pt", VGG16Dialated()),
        ("segnet.pt", SegNet()),
        ("segnet_dialated.pt", SegNetDialated()),
        ("lraspp.pt", LRASPP()),
        ("lraspp_dialated.pt", LRASPPDialated()),
    ]


def main():
    dataset = get_cityscapes_dataset(split="val")
    models_to_check = build_models_to_check()

    print(f"Evaluating with {min(EVAL_SAMPLES, len(dataset))} samples")
    print(f"Device: {DEVICE}\n")

    results = []
    for model_path, model in models_to_check:
        print(model_path)
        try:
            load_model_checkpoint(model, model_path)
            miou = evaluate_miou(model=model, dataset=dataset, num_samples=EVAL_SAMPLES)
            results.append((model_path, miou))
            print(f"mIoU: {miou:.4f}\n")
        except FileNotFoundError as error:
            results.append((model_path, None))
            print(f"{error}\n")

    print("Summary")
    for model_path, miou in results:
        if miou is None:
            print(f"{model_path}: checkpoint missing")
        else:
            print(f"{model_path}: {miou:.4f}")


if __name__ == "__main__":
    main()
