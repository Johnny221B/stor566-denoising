from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parents[1]
RAW_FIELDS = [
    "image_id", "relative_path", "model", "sigma", "seed", "psnr", "ssim",
    "lpips", "runtime_sec", "peak_gpu_mib", "nfe", "status",
    "checkpoint_sha256", "git_commit", "config_path", "output_path",
]


def load_config() -> dict:
    with (ROOT / "configs/experiment.yaml").open() as f:
        return yaml.safe_load(f)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(np.rint(np.clip(image, 0, 1) * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(arr, "RGB").save(path, compress_level=1)


def tensor_from_hwc(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0).to(device)


def hwc_from_tensor(image: torch.Tensor) -> np.ndarray:
    return image.detach().float().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()


def metrics(clean: np.ndarray, output: np.ndarray) -> tuple[float, float]:
    output = np.clip(output.astype(np.float32), 0, 1)
    clean = clean.astype(np.float32)
    psnr = peak_signal_noise_ratio(clean, output, data_range=1.0)
    ssim = structural_similarity(
        clean, output, data_range=1.0, channel_axis=2, gaussian_weights=True,
        sigma=1.5, use_sample_covariance=False,
    )
    return float(psnr), float(ssim)


def manifest_rows() -> list[dict]:
    path = ROOT / "artifacts/test_manifest.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def noisy_path(image_id: str, sigma: int) -> Path:
    return ROOT / "data/noisy" / f"sigma{sigma}" / f"{image_id}.npy"


def output_path(model: str, image_id: str, sigma: int) -> Path:
    return ROOT / "outputs" / model / f"sigma{sigma}" / f"{image_id}.png"


def existing_keys() -> set[tuple[str, str, int]]:
    path = ROOT / "results/raw_metrics.csv"
    if not path.exists():
        return set()
    with path.open(newline="") as f:
        return {(r["image_id"], r["model"], int(r["sigma"])) for r in csv.DictReader(f) if r["status"] == "ok"}


def append_raw(row: dict) -> None:
    path = ROOT / "results/raw_metrics.csv"
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RAW_FIELDS})
        f.flush()

