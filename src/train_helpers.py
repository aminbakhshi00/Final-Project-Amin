import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from tqdm import tqdm

from cityscapes_data import CityscapesDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resize_logits_if_needed(logits, label_shape):
    if logits.shape[-2:] == label_shape:
        return logits

    return F.interpolate(
        logits,
        size=label_shape,
        mode="bilinear",
        align_corners=False,
    )


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs,
    device,
    model_save_path,
    max_epochs_without_save=2,
):
    best_val_loss = float("inf")
    epochs_without_save = 0

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            outputs = resize_logits_if_needed(outputs, labels.shape[-2:])

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / max(len(train_loader), 1)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = model(imgs)
                outputs = resize_logits_if_needed(outputs, labels.shape[-2:])
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

        epoch_val_loss = running_val_loss / max(len(val_loader), 1)
        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Train Loss: {epoch_train_loss:.4f}, "
            f"Validation Loss: {epoch_val_loss:.4f}"
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_without_save = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved model to {model_save_path}")
        else:
            epochs_without_save += 1
            print(
                "Validation loss did not improve, model not saved "
                f"({epochs_without_save}/{max_epochs_without_save})."
            )
            if epochs_without_save >= max_epochs_without_save:
                print(
                    f"No model saved for {max_epochs_without_save} epochs. "
                    "Stopping training."
                )
                break


def compute_iou(pred, target, num_classes, ignore_index):
    valid_mask = target != ignore_index
    pred = pred[valid_mask]
    target = target[valid_mask]

    if pred.size == 0:
        return float("nan")

    ious = []

    for cls in range(num_classes):
        if cls == ignore_index:
            continue

        pred_inds = pred == cls
        target_inds = target == cls

        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()

        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(intersection / union)

    return np.nanmean(ious)


def evaluate_miou(model, dataset, num_samples, num_classes, ignore_index, device):
    model.eval()
    total_iou = []

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            img, label = dataset[i]
            img_input = img.unsqueeze(0).to(device)
            output = model(img_input)
            output = resize_logits_if_needed(output, label.shape[-2:])

            pred = torch.argmax(output[0], dim=0).cpu()
            iou = compute_iou(pred.numpy(), label.numpy(), num_classes, ignore_index)
            total_iou.append(iou)

    if not total_iou:
        print("No samples found for mIoU evaluation.")
        return float("nan")

    mean_iou = np.nanmean(total_iou)
    print(f"Mean IoU over {len(total_iou)} samples: {mean_iou:.4f}")
    return mean_iou


def visualize_predictions(
    model,
    dataset,
    num_samples,
    save_dir,
    num_classes,
    device,
    sample_prefix,
):
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            img, label = dataset[i]
            img_input = img.unsqueeze(0).to(device)
            output = model(img_input)
            output = resize_logits_if_needed(output, label.shape[-2:])

            pred = torch.argmax(output[0], dim=0).cpu().numpy()

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))

            axs[0].imshow(TF.to_pil_image(img))
            axs[0].set_title("Input Image")
            axs[0].axis("off")

            axs[1].imshow(label.numpy(), cmap="tab20", vmin=0, vmax=num_classes - 1)
            axs[1].set_title("Ground Truth")
            axs[1].axis("off")

            axs[2].imshow(pred, cmap="tab20", vmin=0, vmax=num_classes - 1)
            axs[2].set_title("Prediction")
            axs[2].axis("off")

            plt.tight_layout()
            plt.savefig(
                os.path.join(save_dir, f"{sample_prefix}_{i + 1}.pdf"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()


def run_training_pipeline(
    model,
    data_root,
    split,
    num_classes,
    ignore_index,
    batch_size,
    epochs,
    learning_rate,
    num_workers,
    skip_train,
    model_save_path,
    output_dir,
    visualize_samples,
    eval_samples,
    sample_prefix,
):
    train_dataset = CityscapesDataset(
        data_root=data_root,
        split=split,
        num_classes=num_classes,
        ignore_index=ignore_index,
        transform=TF.to_tensor,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_dataset = CityscapesDataset(
        data_root=data_root,
        split="val",
        num_classes=num_classes,
        ignore_index=ignore_index,
        transform=TF.to_tensor,
    )
    if len(val_dataset) == 0:
        print("Validation split is empty. Using training split for validation loss.")
        val_dataset = train_dataset
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if not skip_train:
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=epochs,
            device=DEVICE,
            model_save_path=model_save_path,
        )
    elif os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
        print(f"Loaded model from {model_save_path}")
    else:
        raise FileNotFoundError(
            f"SKIP_TRAIN=True but checkpoint not found: {model_save_path}"
        )

    evaluate_miou(
        model=model,
        dataset=train_dataset,
        num_samples=eval_samples,
        num_classes=num_classes,
        ignore_index=ignore_index,
        device=DEVICE,
    )
    visualize_predictions(
        model=model,
        dataset=train_dataset,
        num_samples=visualize_samples,
        save_dir=output_dir,
        num_classes=num_classes,
        device=DEVICE,
        sample_prefix=sample_prefix,
    )
