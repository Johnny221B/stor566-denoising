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

sys.path.insert(0,str(ROOT/"external/PnP-Flow"))
from pnpflow.models import UNet


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--limit",type=int); ap.add_argument("--device",default="cuda:0"); ap.add_argument("--start",type=int,default=0); args=ap.parse_args()
    cfg=load_config(); device=torch.device(args.device); mcfg=cfg["models"]["pnp_flow"]
    ckpt=ROOT/mcfg["checkpoint"]
    model=UNet(input_channels=3,input_height=256,ch=32,ch_mult=(1,2,4,8),num_res_blocks=6,attn_resolutions=(16,8),resamp_with_conv=True)
    model.load_state_dict(torch.load(ckpt,map_location="cpu",weights_only=True),strict=True); model.eval().requires_grad_(False).to(device)
    digest=sha256(ckpt); commit=subprocess.check_output(["git","-C",str(ROOT/"external/PnP-Flow"),"rev-parse","HEAD"],text=True).strip()
    rows=manifest_rows()[args.start:]
    if args.limit is not None: rows=rows[:args.limit]
    done=existing_keys(); steps=int(mcfg["steps"]); samples=int(mcfg["num_samples"]); delta=1.0/steps
    for sigma in cfg["noise"]["sigmas_255"]:
        sigma_model=2.0*sigma/255.0; base_lr=sigma_model**2*float(mcfg["lr_pnp"])
        for item in rows:
            if (item["image_id"],"pnp_flow",int(sigma)) in done: continue
            image_index=int(item["image_id"].split("_",1)[0]); seed=int(cfg["noise"]["seed_base"])+image_index*1000+int(sigma); seed_everything(seed)
            clean=read_rgb(ROOT/item["clean_path"]); noisy01=np.load(noisy_path(item["image_id"],sigma),allow_pickle=False)
            y=2.0*tensor_from_hwc(noisy01,device)-1.0; x=torch.ones_like(y)
            torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device); start=time.perf_counter()
            with torch.inference_mode():
                for iteration in range(steps):
                    t=torch.full((1,),delta*iteration,device=device); lr=base_lr*(1.0-t).pow(float(mcfg["alpha"])); z=x-lr.view(-1,1,1,1)*(x-y)/(sigma_model**2)
                    x_new=torch.zeros_like(x)
                    for _ in range(samples):
                        z_tilde=t.view(-1,1,1,1)*z+torch.randn_like(z)*(1.0-t.view(-1,1,1,1))
                        x_new += z_tilde+(1.0-t.view(-1,1,1,1))*model(z_tilde,t)
                    x=x_new/samples
            torch.cuda.synchronize(device); elapsed=time.perf_counter()-start; peak=torch.cuda.max_memory_allocated(device)/2**20
            target=output_path("pnp_flow",item["image_id"],sigma); save_rgb(target,hwc_from_tensor((x+1.0)/2.0)); psnr,ssim=metrics(clean,read_rgb(target))
            append_raw(dict(image_id=item["image_id"],relative_path=item["relative_path"],model="pnp_flow",sigma=sigma,seed=seed,psnr=psnr,ssim=ssim,runtime_sec=elapsed,peak_gpu_mib=peak,nfe=steps*samples,status="ok",checkpoint_sha256=digest,git_commit=commit,config_path="configs/experiment.yaml",output_path=target.relative_to(ROOT).as_posix()))
            print(f"pnp_flow sigma={sigma} image={image_index} PSNR={psnr:.3f} time={elapsed:.2f}s",flush=True)


if __name__ == "__main__": main()
