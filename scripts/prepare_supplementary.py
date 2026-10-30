#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from common import ROOT


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs/supplementary.yaml").read_text())
    manifest_dir = ROOT / "artifacts/supplementary"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for dataset, dcfg in cfg["datasets"].items():
        source = ROOT / dcfg["source_root"]
        paths = sorted(p for p in source.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
        paths = paths[: int(dcfg["count"])]
        if len(paths) != int(dcfg["count"]):
            raise RuntimeError(f"{dataset}: expected {dcfg['count']} images, found {len(paths)}")
        rows = []
        for index, path in enumerate(paths):
            image_id = f"{index:03d}_{path.stem}"
            clean_path = ROOT / "data/supplementary/clean" / dataset / f"{image_id}.png"
            image = Image.open(path).convert("RGB").resize(tuple(cfg["resolution"]), Image.Resampling.BICUBIC)
            clean_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(clean_path)
            clean = np.asarray(image, dtype=np.float32) / 255.0
            digest = hashlib.sha256(clean_path.read_bytes()).hexdigest()
            rows.append({"image_id": image_id, "source_name": path.name,
                         "clean_path": clean_path.relative_to(ROOT).as_posix(), "sha256": digest})
            for noise_type in dcfg["noise_types"]:
                for sigma in cfg["sigmas_255"]:
                    seed = int(cfg["seed_base"]) + index * 1000 + int(sigma) + (100000 if noise_type == "laplace" else 0)
                    rng = np.random.default_rng(seed)
                    std = sigma / 255.0
                    noise = (rng.normal(0.0, std, clean.shape) if noise_type == "gaussian"
                             else rng.laplace(0.0, std / np.sqrt(2.0), clean.shape))
                    noisy = np.clip(clean + noise, 0.0, 1.0).astype(np.float32)
                    target = ROOT / "data/supplementary/noisy" / dataset / noise_type / f"sigma{sigma}" / f"{image_id}.npy"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    np.save(target, noisy, allow_pickle=False)
        output = manifest_dir / f"{dataset}.jsonl"
        output.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        print(f"{dataset}: wrote {len(rows)} clean images and manifest {output}")


if __name__ == "__main__":
    main()
