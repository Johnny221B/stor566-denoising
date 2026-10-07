#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from common import ROOT, manifest_rows, metrics, read_rgb


def main() -> None:
    raw=pd.read_csv(ROOT/"results/raw_metrics.csv")
    baseline=pd.DataFrame(json.loads(x) for x in (ROOT/"results/noisy_baseline.jsonl").read_text().splitlines())
    baseline["model"]="noisy"; baseline["runtime_sec"]=np.nan; baseline["peak_gpu_mib"]=np.nan; baseline["nfe"]=np.nan; baseline["status"]="ok"
    combined=pd.concat([raw,baseline],ignore_index=True,sort=False)
    summary=combined.groupby(["model","sigma"],as_index=False).agg(attempted=("image_id","count"),succeeded=("status",lambda s:(s=="ok").sum()),failed=("status",lambda s:(s!="ok").sum()),psnr_mean=("psnr","mean"),psnr_std=("psnr","std"),ssim_mean=("ssim","mean"),ssim_std=("ssim","std"),runtime_median=("runtime_sec","median"),runtime_mean=("runtime_sec","mean"),peak_gpu_mib_max=("peak_gpu_mib","max"),nfe_mean=("nfe","mean"))
    summary.to_csv(ROOT/"results/summary.csv",index=False)
    quality=summary[["model","sigma","attempted","psnr_mean","psnr_std","ssim_mean","ssim_std"]].copy(); quality.to_csv(ROOT/"artifacts/tables/main_quality.csv",index=False)
    summary[["model","sigma","runtime_median","runtime_mean","peak_gpu_mib_max","nfe_mean"]].to_csv(ROOT/"artifacts/tables/efficiency.csv",index=False)
    for metric,label in [("psnr_mean","PSNR (dB)"),("ssim_mean","SSIM")]:
        fig,ax=plt.subplots(figsize=(7,4.5))
        for model,g in summary.groupby("model"):
            ax.plot(g.sigma,g[metric],marker="o",label=model)
        ax.set(xlabel="Gaussian noise σ (0–255 scale)",ylabel=label,xticks=[15,25,50]); ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(ROOT/f"artifacts/plots/{'psnr' if metric.startswith('psnr') else 'ssim'}_vs_sigma.png",dpi=180); plt.close(fig)
    timed=summary[summary.model!="noisy"]
    fig,ax=plt.subplots(figsize=(7,4.5))
    for _,r in timed.iterrows(): ax.scatter(r.runtime_mean,r.psnr_mean,label=f"{r.model} σ={int(r.sigma)}")
    ax.set(xlabel="Mean inference time (seconds/image, log scale)",ylabel="Mean PSNR (dB)",xscale="log"); ax.grid(alpha=.3); ax.legend(fontsize=7,ncol=2); fig.tight_layout(); fig.savefig(ROOT/"artifacts/plots/quality_runtime_tradeoff.png",dpi=180); plt.close(fig)

    chosen=[0,1,2,3,4,5]; models=["dncnn","drunet","diffusion","pnp_flow"]
    manifest=manifest_rows()
    for sigma in [15,25,50]:
        for idx in chosen:
            item=manifest[idx]; clean=Image.open(ROOT/item["clean_path"]).convert("RGB"); noisy=Image.open(ROOT/"data/noisy"/f"sigma{sigma}"/f"{item['image_id']}.png").convert("RGB")
            ims=[clean,noisy]+[Image.open(ROOT/"outputs"/m/f"sigma{sigma}"/f"{item['image_id']}.png").convert("RGB") for m in models]
            labels=["Clean","Noisy","DnCNN","DRUNet","DiffPIR","PnP-Flow"]
            fig,axes=plt.subplots(1,6,figsize=(18,3.25));
            for ax,im,label in zip(axes,ims,labels): ax.imshow(im); ax.set_title(label); ax.axis("off")
            fig.tight_layout(); fig.savefig(ROOT/"artifacts/qualitative"/f"{item['image_id']}_sigma{sigma}_grid.png",dpi=150); plt.close(fig)
            clean_np=np.asarray(clean,dtype=np.float32)/255
            errors=[np.abs(clean_np-np.asarray(im,dtype=np.float32)/255).mean(2) for im in ims[1:]]; vmax=max(float(e.max()) for e in errors)
            fig,axes=plt.subplots(1,5,figsize=(15,3.25));
            for ax,e,label in zip(axes,errors,labels[1:]): q=ax.imshow(e,cmap="magma",vmin=0,vmax=vmax); ax.set_title(label); ax.axis("off")
            fig.colorbar(q,ax=axes.tolist(),shrink=.75,label="Mean absolute RGB error"); fig.savefig(ROOT/"artifacts/qualitative"/f"{item['image_id']}_sigma{sigma}_errors.png",dpi=150,bbox_inches="tight"); plt.close(fig)
    print(summary.to_string(index=False))


if __name__=="__main__": main()

