#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from common import ROOT, load_config, metrics, save_rgb


def main() -> None:
    cfg = load_config()
    source = ROOT / cfg["dataset"]["source_root"]
    paths = sorted(p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    paths = paths[: int(cfg["dataset"]["num_test_images"])]
    if len(paths) != 100 or len(set(paths)) != len(paths):
        raise RuntimeError(f"Expected 100 unique test images, got {len(paths)}")

    manifest = []
    baseline = []
    for index, src in enumerate(paths):
        image_id = f"{index:03d}_{src.stem}"
        relative = src.relative_to(ROOT / "external/PnP-Flow/data/afhq_cat").as_posix()
        clean_path = ROOT / "data/clean" / f"{image_id}.png"
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        clean_u8 = np.asarray(Image.open(src).convert("RGB").resize((256, 256), resampling), dtype=np.uint8)
        clean = clean_u8.astype(np.float32) / 255.0
        save_rgb(clean_path, clean)
        entry = {"image_id": image_id, "relative_path": relative, "clean_path": clean_path.relative_to(ROOT).as_posix()}
        manifest.append(entry)
        for sigma in cfg["noise"]["sigmas_255"]:
            seed = int(cfg["noise"]["seed_base"]) + index * 1000 + int(sigma)
            rng = np.random.default_rng(seed)
            noisy = np.clip(clean + rng.normal(0, sigma / 255.0, clean.shape).astype(np.float32), 0, 1).astype(np.float32)
            npy = ROOT / "data/noisy" / f"sigma{sigma}" / f"{image_id}.npy"
            png = ROOT / "data/noisy" / f"sigma{sigma}" / f"{image_id}.png"
            npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(npy, noisy, allow_pickle=False)
            save_rgb(png, noisy)
            psnr, ssim = metrics(clean, noisy)
            baseline.append({"image_id": image_id, "relative_path": relative, "sigma": sigma, "seed": seed, "psnr": psnr, "ssim": ssim, "npy": npy.relative_to(ROOT).as_posix()})

    (ROOT / "artifacts/test_manifest.txt").write_text("\n".join(x["relative_path"] for x in manifest) + "\n")
    with (ROOT / "artifacts/test_manifest.jsonl").open("w") as f:
        for x in manifest:
            f.write(json.dumps(x) + "\n")
    with (ROOT / "results/noisy_baseline.jsonl").open("w") as f:
        for x in baseline:
            f.write(json.dumps(x) + "\n")
    print(f"Prepared {len(manifest)} clean and {len(baseline)} noisy tensors")


if __name__ == "__main__":
    main()

