#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import deepinv as dinv
import h5py
import numpy as np
import torch
import yaml
from torch import nn

from common import ROOT, hwc_from_tensor, metrics, read_rgb, save_rgb, seed_everything, sha256, tensor_from_hwc

sys.path.insert(0, str(ROOT / "external/DPIR"))
from models.network_unet import UNetRes
sys.path.insert(0, str(ROOT / "external/PnP-Flow"))
from pnpflow.models import UNet
from pnpflow.methods.pnp_diff import L1


def decode(dataset) -> str:
    return "".join(chr(int(x)) for x in np.asarray(dataset).reshape(-1))


class MatConvNetDnCNN(nn.Module):
    def __init__(self, checkpoint: Path):
        super().__init__(); layers = []
        with h5py.File(checkpoint, "r") as f:
            for ref in f["net/layers"][:, 0]:
                layer = f[ref]; kind = decode(layer["type"])
                if kind == "conv":
                    refs = layer["weights"][:, 0]; weight = np.asarray(f[refs[0]], dtype=np.float32); bias = np.asarray(f[refs[1]], dtype=np.float32).reshape(-1)
                    conv = nn.Conv2d(weight.shape[1], weight.shape[0], 3, padding=1); conv.weight.data.copy_(torch.from_numpy(weight)); conv.bias.data.copy_(torch.from_numpy(bias)); layers.append(conv)
                elif kind == "relu": layers.append(nn.ReLU(inplace=True))
                else: raise ValueError(kind)
        self.net = nn.Sequential(*layers)
    def forward(self, x): return x - self.net(x)


def load_model(name, sigma, cfg, device):
    if name == "dncnn":
        ckpt = ROOT / cfg["models"][name]["checkpoints"][int(sigma)]; model = MatConvNetDnCNN(ckpt)
    elif name == "drunet":
        ckpt = ROOT / cfg["models"][name]["checkpoint"]
        model = UNetRes(in_nc=4, out_nc=3, nc=[64,128,256,512], nb=4, act_mode="R", downsample_mode="strideconv", upsample_mode="convtranspose")
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
    elif name == "diffusion":
        ckpt = ROOT / cfg["models"][name]["checkpoint"]; model = dinv.models.DiffUNet(pretrained=str(ckpt))
    else:
        ckpt = ROOT / cfg["models"][name]["checkpoint"]
        model = UNet(input_channels=3,input_height=256,ch=32,ch_mult=(1,2,4,8),num_res_blocks=6,attn_resolutions=(16,8),resamp_with_conv=True)
        model.load_state_dict(torch.load(ckpt,map_location="cpu",weights_only=True),strict=True)
    return model.eval().requires_grad_(False).to(device), ckpt


def infer(name, model, y01, sigma, noise_type, cfg, device):
    if name == "dncnn": return model(y01), 1
    if name == "drunet":
        noise_map = torch.full((1,1,256,256), sigma/255.0, device=device); return model(torch.cat((y01, noise_map), 1)), 1
    if name == "diffusion":
        sigma01 = sigma / 255.0; physics = dinv.physics.Denoising(noise_model=dinv.physics.GaussianNoise(sigma=sigma01), device=device)
        fidelity = dinv.optim.L2() if noise_type == "gaussian" else L1(); mcfg = cfg["models"][name]
        solver = dinv.sampling.DiffPIR(model, data_fidelity=fidelity, sigma=sigma01, zeta=mcfg["zeta"], lambda_=mcfg["lambda"], device=device)
        return solver(y01, physics), 100
    mcfg = cfg["models"][name]; steps = int(mcfg["steps"]); samples = int(mcfg["num_samples"]); y = 2.0*y01-1.0; x = torch.ones_like(y); delta = 1.0/steps
    if noise_type == "gaussian": scale = 2.0*sigma/255.0; base_lr = scale**2*float(mcfg["lr_pnp"])
    else: scale = 2.0*(sigma/255.0)/np.sqrt(2.0); base_lr = scale*float(mcfg["lr_pnp"])
    for iteration in range(steps):
        t = torch.full((1,), delta*iteration, device=device); lr = base_lr*(1.0-t).pow(float(mcfg["alpha"])); grad = (x-y)/(scale**2) if noise_type == "gaussian" else torch.sign(x-y)/scale
        z = x-lr.view(-1,1,1,1)*grad; x_new = torch.zeros_like(x)
        for _ in range(samples):
            tv=t.view(-1,1,1,1); zt=tv*z+torch.randn_like(z)*(1.0-tv); x_new += zt+(1.0-tv)*model(zt,t)
        x=x_new/samples
    return (x+1.0)/2.0, steps*samples


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",required=True); ap.add_argument("--noise",required=True,choices=["gaussian","laplace"]); ap.add_argument("--model",required=True,choices=["dncnn","drunet","diffusion","pnp_flow"]); ap.add_argument("--device",default="cuda:0"); ap.add_argument("--start",type=int,default=0); ap.add_argument("--limit",type=int); ap.add_argument("--sigma",type=int,choices=[15,25,50]); args=ap.parse_args()
    cfg=yaml.safe_load((ROOT/"configs/supplementary.yaml").read_text()); device=torch.device(args.device); rows=[json.loads(x) for x in (ROOT/f"artifacts/supplementary/{args.dataset}.jsonl").read_text().splitlines() if x.strip()][args.start:]
    if args.limit is not None: rows=rows[:args.limit]
    commits={"dncnn":"external/DnCNN","drunet":"external/DPIR","diffusion":"external/PnP-Flow","pnp_flow":"external/PnP-Flow"}; commit=subprocess.check_output(["git","-C",str(ROOT/commits[args.model]),"rev-parse","HEAD"],text=True).strip()
    for sigma in ([args.sigma] if args.sigma is not None else cfg["sigmas_255"]):
        model,ckpt=load_model(args.model,sigma,cfg,device); digest=sha256(ckpt)
        for pos,item in enumerate(rows):
            recpath=ROOT/f"results/supplementary_records/{args.dataset}/{args.noise}/{args.model}/sigma{sigma}/{item['image_id']}.json"
            if recpath.exists(): continue
            index=args.start+pos; seed=int(cfg["seed_base"])+index*1000+int(sigma)+(100000 if args.noise=="laplace" else 0); seed_everything(seed); clean=read_rgb(ROOT/item["clean_path"]); noisy=np.load(ROOT/f"data/supplementary/noisy/{args.dataset}/{args.noise}/sigma{sigma}/{item['image_id']}.npy",allow_pickle=False); y=tensor_from_hwc(noisy,device)
            torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device); started=time.perf_counter()
            with torch.inference_mode(): out,nfe=infer(args.model,model,y,sigma,args.noise,cfg,device)
            torch.cuda.synchronize(device); elapsed=time.perf_counter()-started; peak=torch.cuda.max_memory_allocated(device)/2**20; target=ROOT/f"outputs/supplementary/{args.dataset}/{args.noise}/{args.model}/sigma{sigma}/{item['image_id']}.png"; save_rgb(target,hwc_from_tensor(out)); psnr,ssim=metrics(clean,read_rgb(target))
            record=dict(dataset=args.dataset,noise_type=args.noise,image_id=item["image_id"],source_name=item["source_name"],model=args.model,sigma=sigma,seed=seed,psnr=psnr,ssim=ssim,runtime_sec=elapsed,peak_gpu_mib=peak,nfe=nfe,status="ok",checkpoint_sha256=digest,source_commit=commit,output_path=target.relative_to(ROOT).as_posix()); recpath.parent.mkdir(parents=True,exist_ok=True); recpath.write_text(json.dumps(record,sort_keys=True)+"\n"); print(f"{args.dataset}/{args.noise} {args.model} sigma={sigma} {pos+1}/{len(rows)} PSNR={psnr:.3f}",flush=True)


if __name__=="__main__": main()
