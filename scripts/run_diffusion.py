#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import time

import deepinv as dinv
import numpy as np
import torch

from common import (ROOT, append_raw, existing_keys, hwc_from_tensor, load_config,
                    manifest_rows, metrics, noisy_path, output_path, read_rgb,
                    save_rgb, seed_everything, sha256, tensor_from_hwc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(); device = torch.device(args.device)
    ckpt = ROOT / "checkpoints/diffusion_ffhq_10m.pt"
    model = dinv.models.DiffUNet(pretrained=str(ckpt)).eval().requires_grad_(False).to(device)
    digest = sha256(ckpt)
    commit = subprocess.check_output(["git", "-C", str(ROOT / "external/PnP-Flow"), "rev-parse", "HEAD"], text=True).strip()
    rows = manifest_rows()[args.start:]
    if args.limit is not None: rows = rows[:args.limit]
    done = existing_keys()
    for sigma in cfg["noise"]["sigmas_255"]:
        sigma01 = sigma / 255.0
        physics = dinv.physics.Denoising(noise_model=dinv.physics.GaussianNoise(sigma=sigma01), device=device)
        solver = dinv.sampling.DiffPIR(model, data_fidelity=dinv.optim.L2(), sigma=sigma01,
            zeta=cfg["models"]["diffusion"]["zeta"], lambda_=cfg["models"]["diffusion"]["lambda"], device=device)
        for item in rows:
            if (item["image_id"], "diffusion", int(sigma)) in done: continue
            image_index=int(item["image_id"].split("_",1)[0]); seed=int(cfg["noise"]["seed_base"])+image_index*1000+int(sigma)
            seed_everything(seed); clean=read_rgb(ROOT/item["clean_path"]); noisy=np.load(noisy_path(item["image_id"],sigma),allow_pickle=False)
            y=tensor_from_hwc(noisy,device)
            torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device); start=time.perf_counter()
            with torch.inference_mode(): out_t=solver(y,physics)
            torch.cuda.synchronize(device); elapsed=time.perf_counter()-start
            peak=torch.cuda.max_memory_allocated(device)/2**20; target=output_path("diffusion",item["image_id"],sigma); save_rgb(target,hwc_from_tensor(out_t))
            psnr,ssim=metrics(clean,read_rgb(target))
            append_raw(dict(image_id=item["image_id"],relative_path=item["relative_path"],model="diffusion",sigma=sigma,seed=seed,psnr=psnr,ssim=ssim,runtime_sec=elapsed,peak_gpu_mib=peak,nfe=100,status="ok",checkpoint_sha256=digest,git_commit=commit,config_path="configs/experiment.yaml",output_path=target.relative_to(ROOT).as_posix()))
            print(f"diffusion sigma={sigma} image={image_index} PSNR={psnr:.3f} time={elapsed:.2f}s",flush=True)


if __name__ == "__main__": main()

