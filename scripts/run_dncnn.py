#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from common import (ROOT, append_raw, existing_keys, hwc_from_tensor, load_config,
                    manifest_rows, metrics, noisy_path, output_path, read_rgb,
                    save_rgb, seed_everything, sha256, tensor_from_hwc)


def decode(dataset) -> str:
    return "".join(chr(int(x)) for x in np.asarray(dataset).reshape(-1))


class MatConvNetDnCNN(nn.Module):
    def __init__(self, checkpoint: Path):
        super().__init__()
        layers = []
        with h5py.File(checkpoint, "r") as f:
            for ref in f["net/layers"][:, 0]:
                layer = f[ref]
                kind = decode(layer["type"])
                if kind == "conv":
                    refs = layer["weights"][:, 0]
                    weight = np.asarray(f[refs[0]], dtype=np.float32)
                    bias = np.asarray(f[refs[1]], dtype=np.float32).reshape(-1)
                    conv = nn.Conv2d(weight.shape[1], weight.shape[0], 3, padding=1, bias=True)
                    conv.weight.data.copy_(torch.from_numpy(weight))
                    conv.bias.data.copy_(torch.from_numpy(bias))
                    layers.append(conv)
                elif kind == "relu":
                    layers.append(nn.ReLU(inplace=True))
                else:
                    raise ValueError(f"Unsupported MatConvNet layer {kind}")
        self.net = nn.Sequential(*layers)

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        return noisy - self.net(noisy)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    cfg = load_config()
    device = torch.device(args.device)
    rows = manifest_rows()[: args.limit]
    done = existing_keys()
    commit = subprocess.check_output(["git", "-C", str(ROOT / "external/DnCNN"), "rev-parse", "HEAD"], text=True).strip()
    for sigma in cfg["noise"]["sigmas_255"]:
        ckpt = ROOT / cfg["models"]["dncnn"]["checkpoints"][int(sigma)]
        digest = sha256(ckpt)
        model = MatConvNetDnCNN(ckpt).eval().to(device)
        model.requires_grad_(False)
        dummy = torch.zeros((1, 3, 256, 256), device=device)
        with torch.inference_mode():
            for _ in range(cfg["evaluation"]["warmup"]): model(dummy)
        for index, item in enumerate(rows):
            key = (item["image_id"], "dncnn", int(sigma))
            if key in done: continue
            seed = int(cfg["noise"]["seed_base"]) + int(item["image_id"].split("_", 1)[0]) * 1000 + int(sigma)
            seed_everything(seed)
            clean = read_rgb(ROOT / item["clean_path"])
            noisy = np.load(noisy_path(item["image_id"], sigma), allow_pickle=False)
            x = tensor_from_hwc(noisy, device)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device); start = time.perf_counter()
            with torch.inference_mode(): out_t = model(x)
            torch.cuda.synchronize(device); elapsed = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated(device) / 2**20
            out = hwc_from_tensor(out_t)
            target = output_path("dncnn", item["image_id"], sigma)
            save_rgb(target, out)
            saved = read_rgb(target)
            psnr, ssim = metrics(clean, saved)
            append_raw(dict(image_id=item["image_id"], relative_path=item["relative_path"], model="dncnn", sigma=sigma, seed=seed, psnr=psnr, ssim=ssim, runtime_sec=elapsed, peak_gpu_mib=peak, status="ok", checkpoint_sha256=digest, git_commit=commit, config_path="configs/experiment.yaml", output_path=target.relative_to(ROOT).as_posix()))
            print(f"dncnn sigma={sigma} {index+1}/{len(rows)} PSNR={psnr:.3f}")


if __name__ == "__main__": main()

