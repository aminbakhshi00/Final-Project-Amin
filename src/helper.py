import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import DATA_ROOT, IGNORE_INDEX, get_cityscapes_dataset

DEVICE = torch.device("cuda")
def train_model(model, train_loader, val_loader, epochs, learning_rate, model_path):
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                logits = model(images)
                val_loss += criterion(logits, labels).item()

        train_loss /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        print(f"Epoch {epoch + 1}: train={train_loss:.4f} val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"Saved model to {model_path}")


def load_model_checkpoint(model, model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def compute_iou(pred, target):
    valid_mask = target != IGNORE_INDEX
    pred = pred[valid_mask]
    target = target[valid_mask]

    if pred.size == 0:
        return float("nan")

    ious = []
    for class_id in range(19):
        pred_mask = pred == class_id
        target_mask = target == class_id
        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(intersection / union)

    return np.nanmean(ious)


def evaluate_miou(model, dataset, num_samples=100):
    model.eval()
    ious = []

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            image, label = dataset[i]
            logits = model(image.unsqueeze(0).to(DEVICE))
            pred = torch.argmax(logits[0], dim=0).cpu().numpy()
            iou = compute_iou(pred, label.numpy())
            ious.append(iou)

    if not ious:
        print("No samples found for mIoU evaluation.")
        return float("nan")

    mean_iou = np.nanmean(ious)
    print(f"Mean IoU over {len(ious)} samples: {mean_iou:.4f}")
    return mean_iou


def visualize_predictions(model, dataset, save_dir, sample_prefix, num_samples=3):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            image, label = dataset[i]
            logits = model(image.unsqueeze(0).to(DEVICE))
            pred = torch.argmax(logits[0], dim=0).cpu().numpy()

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(TF.to_pil_image(image))
            axes[0].set_title("Input")
            axes[0].axis("off")

            axes[1].imshow(label.numpy(), cmap="tab20", vmin=0, vmax=18)
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            axes[2].imshow(pred, cmap="tab20", vmin=0, vmax=18)
            axes[2].set_title("Prediction")
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig(
                os.path.join(save_dir, f"{sample_prefix}_{i + 1}.pdf"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()


def run_training_pipeline(
    model,
    model_name,
    batch_size=30,
    epochs=15,
    learning_rate=1e-4,
    num_workers=4,
    skip_train=False,
    data_root=DATA_ROOT,
    eval_samples=100,
    visualize_samples=3,
):
    train_dataset = get_cityscapes_dataset(data_root=data_root, split="train")
    val_dataset = get_cityscapes_dataset(data_root=data_root, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = model.to(DEVICE)
    model_path = f"{model_name}.pt"

    if skip_train:
        load_model_checkpoint(model, model_path)
        print(f"Loaded model from {model_path}")
    else:
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            model_path=model_path,
        )
        load_model_checkpoint(model, model_path)

    evaluate_miou(model, val_dataset, num_samples=eval_samples)
    visualize_predictions(
        model=model,
        dataset=val_dataset,
        save_dir=os.path.join("outputs", model_name),
        sample_prefix=model_name,
        num_samples=visualize_samples,
    )
