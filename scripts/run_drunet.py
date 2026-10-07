#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time

import numpy as np
import torch

from common import (ROOT, append_raw, existing_keys, hwc_from_tensor, load_config,
                    manifest_rows, metrics, noisy_path, output_path, read_rgb,
                    save_rgb, seed_everything, sha256, tensor_from_hwc)

sys.path.insert(0, str(ROOT / "external/DPIR"))
from models.network_unet import UNetRes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    cfg = load_config(); device = torch.device(args.device)
    ckpt = ROOT / cfg["models"]["drunet"]["checkpoint"]
    model = UNetRes(in_nc=4, out_nc=3, nc=[64,128,256,512], nb=4, act_mode="R", downsample_mode="strideconv", upsample_mode="convtranspose")
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
    model.eval().requires_grad_(False).to(device)
    digest = sha256(ckpt)
    commit = subprocess.check_output(["git", "-C", str(ROOT / "external/DPIR"), "rev-parse", "HEAD"], text=True).strip()
    rows = manifest_rows()[: args.limit]; done = existing_keys()
    dummy = torch.zeros((1,4,256,256), device=device)
    with torch.inference_mode():
        for _ in range(cfg["evaluation"]["warmup"]): model(dummy)
    for sigma in cfg["noise"]["sigmas_255"]:
        for index, item in enumerate(rows):
            key = (item["image_id"], "drunet", int(sigma))
            if key in done: continue
            seed = int(cfg["noise"]["seed_base"]) + int(item["image_id"].split("_",1)[0])*1000 + int(sigma)
            seed_everything(seed)
            clean = read_rgb(ROOT / item["clean_path"])
            noisy = np.load(noisy_path(item["image_id"], sigma), allow_pickle=False)
            x = tensor_from_hwc(noisy, device)
            noise_map = torch.full((1,1,x.shape[2],x.shape[3]), sigma/255.0, device=device)
            inp = torch.cat((x, noise_map), dim=1)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device); start=time.perf_counter()
            with torch.inference_mode(): out_t=model(inp)
            torch.cuda.synchronize(device); elapsed=time.perf_counter()-start
            peak=torch.cuda.max_memory_allocated(device)/2**20
            out=hwc_from_tensor(out_t); target=output_path("drunet",item["image_id"],sigma)
            save_rgb(target,out); saved=read_rgb(target); psnr,ssim=metrics(clean,saved)
            append_raw(dict(image_id=item["image_id"],relative_path=item["relative_path"],model="drunet",sigma=sigma,seed=seed,psnr=psnr,ssim=ssim,runtime_sec=elapsed,peak_gpu_mib=peak,status="ok",checkpoint_sha256=digest,git_commit=commit,config_path="configs/experiment.yaml",output_path=target.relative_to(ROOT).as_posix()))
            print(f"drunet sigma={sigma} {index+1}/{len(rows)} PSNR={psnr:.3f}")


if __name__ == "__main__": main()
