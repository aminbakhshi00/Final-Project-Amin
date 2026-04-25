#!/usr/bin/env python3
"""Measure model parameter counts and inference time over validation samples."""

import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
OUTPUT_PATH = ROOT_DIR / "outputs" / "inference_time_and_parameters.txt"
NUM_SAMPLES = 10
WARMUP_STEPS = 2

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataloader import get_cityscapes_dataset  # noqa: E402
from evaluate_all_miou import build_models_to_check  # noqa: E402


def synchronize_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


@torch.inference_mode()
def time_model_inference(model, cpu_images, device):
    model = model.to(device).eval()
    images = [image.to(device) for image in cpu_images]

    for _ in range(WARMUP_STEPS):
        _ = model(images[0])
    synchronize_if_cuda(device)

    sample_times = []
    for image in images:
        synchronize_if_cuda(device)
        start = perf_counter()
        _ = model(image)
        synchronize_if_cuda(device)
        sample_times.append(perf_counter() - start)

    return sample_times


def try_time_model(model, cpu_images, preferred_device):
    try:
        sample_times = time_model_inference(model, cpu_images, preferred_device)
        return sample_times, preferred_device
    except RuntimeError as error:
        if preferred_device.type != "cuda" or "out of memory" not in str(error).lower():
            raise
        torch.cuda.empty_cache()
        cpu_device = torch.device("cpu")
        sample_times = time_model_inference(model.cpu(), cpu_images, cpu_device)
        return sample_times, cpu_device


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    preferred_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preferred_device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    dataset = get_cityscapes_dataset(split="val")
    sample_count = min(NUM_SAMPLES, len(dataset))
    if sample_count == 0:
        raise RuntimeError("Validation dataset is empty.")

    cpu_images = []
    for index in range(sample_count):
        image, _ = dataset[index]
        cpu_images.append(image.unsqueeze(0).contiguous())

    _, _, height, width = cpu_images[0].shape
    results = []

    for model_path, model in build_models_to_check():
        checkpoint_path = ROOT_DIR / model_path
        if not checkpoint_path.exists():
            results.append(
                {
                    "model": model_path,
                    "missing": True,
                    "message": f"Checkpoint not found: {checkpoint_path}",
                }
            )
            continue

        state_dict = torch.load(checkpoint_path, map_location=preferred_device)
        model.load_state_dict(state_dict)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        sample_times, used_device = try_time_model(
            model=model, cpu_images=cpu_images, preferred_device=preferred_device
        )
        total_time = sum(sample_times)
        average_time = total_time / len(sample_times)

        results.append(
            {
                "model": model_path,
                "missing": False,
                "parameters": parameter_count,
                "device": str(used_device),
                "total_time": total_time,
                "average_time": average_time,
                "sample_times": sample_times,
            }
        )

        del model
        if preferred_device.type == "cuda":
            torch.cuda.empty_cache()

    lines = [
        "Inference Time and Parameter Report",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Validation samples per model: {sample_count}",
        f"Input resolution used: {height}x{width}",
        f"Preferred device: {preferred_device}",
        f"Warmup iterations per model: {WARMUP_STEPS}",
        "",
    ]

    for result in results:
        lines.append(f"Model: {result['model']}")
        if result["missing"]:
            lines.append(result["message"])
            lines.append("")
            continue

        lines.append(f"Number of parameters: {result['parameters']:,}")
        lines.append(f"Device used for timing: {result['device']}")
        lines.append(
            f"Estimated inference time for {sample_count} samples (seconds): "
            f"{result['total_time']:.4f}"
        )
        lines.append(
            f"Average inference time per sample (seconds): {result['average_time']:.4f}"
        )
        lines.append(
            "Per-sample inference times (seconds): "
            + ", ".join(f"{value:.4f}" for value in result["sample_times"])
        )
        lines.append("")

    OUTPUT_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote report to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
