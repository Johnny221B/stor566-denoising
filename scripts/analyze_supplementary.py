#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import wilcoxon

from common import ROOT

MODELS = ["dncnn", "drunet", "diffusion", "pnp_flow"]


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues); adjusted = np.empty(len(pvalues)); running = 0.0; m = len(pvalues)
    for rank, idx in enumerate(order):
        running = max(running, (m-rank)*pvalues[idx]); adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def main() -> None:
    (ROOT / "figures").mkdir(parents=True, exist_ok=True)
    paths = sorted((ROOT / "results/supplementary_records").glob("**/*.json"))
    records = [json.loads(p.read_text()) for p in paths]
    raw = pd.DataFrame(records).sort_values(["dataset","noise_type","sigma","model","image_id"])
    expected = (100+68)*3*4
    if len(raw) != expected or raw.duplicated(["dataset","noise_type","sigma","model","image_id"]).any():
        raise RuntimeError(f"Expected {expected} unique records, found {len(raw)}")
    raw.to_csv(ROOT / "results/supplementary_raw_metrics.csv", index=False)

    rng = np.random.default_rng(566)
    summary=[]
    for keys,g in raw.groupby(["dataset","noise_type","sigma","model"],sort=True):
        row=dict(zip(["dataset","noise_type","sigma","model"],keys)); row["n"]=len(g)
        for metric in ["psnr","ssim","runtime_sec","peak_gpu_mib"]:
            x=g[metric].to_numpy(); row[f"{metric}_mean"]=x.mean(); row[f"{metric}_sd"]=x.std(ddof=1)
            means=rng.choice(x,(10000,len(x)),replace=True).mean(1); row[f"{metric}_ci_low"],row[f"{metric}_ci_high"]=np.quantile(means,[.025,.975])
        summary.append(row)
    pd.DataFrame(summary).to_csv(ROOT/"results/supplementary_summary.csv",index=False)

    tests=[]
    for keys,g in raw.groupby(["dataset","noise_type","sigma"],sort=True):
        for metric in ["psnr","ssim"]:
            local=[]
            pivot=g.pivot(index="image_id",columns="model",values=metric)
            for a,b in itertools.combinations(MODELS,2):
                diff=pivot[a]-pivot[b]; stat,p=wilcoxon(diff,zero_method="pratt",alternative="two-sided")
                local.append(dict(dataset=keys[0],noise_type=keys[1],sigma=keys[2],metric=metric,model_a=a,model_b=b,n=len(diff),mean_difference=diff.mean(),median_difference=diff.median(),wilcoxon_statistic=stat,p_value=p))
            adj=holm([x["p_value"] for x in local])
            for row,padj in zip(local,adj): row["p_holm"]=padj; tests.append(row)
    pd.DataFrame(tests).to_csv(ROOT/"results/supplementary_paired_tests.csv",index=False)

    fig,axes=plt.subplots(2,3,figsize=(12,6.5),sharex=True)
    for ridx,(dataset,noise) in enumerate([("afhq_cat","laplace"),("cbsd68","gaussian")]):
        sub=raw[(raw.dataset==dataset)&(raw.noise_type==noise)]
        for cidx,metric in enumerate(["psnr","ssim","runtime_sec"]):
            ax=axes[ridx,cidx]
            for model in MODELS:
                means=sub[sub.model==model].groupby("sigma")[metric].mean(); ax.plot(means.index,means.values,marker="o",label=model)
            ax.set_title(f"{dataset}: {noise} — {metric}"); ax.set_xlabel(r"$\sigma_{eq}$ (8-bit)"); ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=8); fig.tight_layout(); fig.savefig(ROOT/"figures/supplementary_comparison.png",dpi=180); plt.close(fig)

    # Case grid: median-PSNR image for each condition at sigma=50.
    fig,axes=plt.subplots(2,6,figsize=(15,5.2))
    for rowidx,(dataset,noise) in enumerate([("afhq_cat","laplace"),("cbsd68","gaussian")]):
        sub=raw[(raw.dataset==dataset)&(raw.noise_type==noise)&(raw.sigma==50)]
        avg=sub.groupby("image_id").psnr.mean().sort_values(); image_id=avg.index[len(avg)//2]
        item=sub[sub.image_id==image_id].iloc[0]; clean=np.asarray(Image.open(ROOT/f"data/supplementary/clean/{dataset}/{image_id}.png")); noisy=np.load(ROOT/f"data/supplementary/noisy/{dataset}/{noise}/sigma50/{image_id}.npy")
        axes[rowidx,0].imshow(clean); axes[rowidx,0].set_title(f"{dataset}\nclean")
        axes[rowidx,1].imshow(np.clip(noisy,0,1)); axes[rowidx,1].set_title(f"{noise} noisy")
        for j,model in enumerate(MODELS,2):
            r=sub[(sub.image_id==image_id)&(sub.model==model)].iloc[0]; axes[rowidx,j].imshow(Image.open(ROOT/r.output_path)); axes[rowidx,j].set_title(f"{model}\n{r.psnr:.2f} dB / {r.ssim:.3f}")
        for ax in axes[rowidx]: ax.axis("off")
    fig.tight_layout(); fig.savefig(ROOT/"figures/supplementary_cases.png",dpi=180); plt.close(fig)
    print(f"wrote {len(raw)} raw rows, {len(summary)} summary rows, {len(tests)} paired tests")


if __name__ == "__main__": main()
