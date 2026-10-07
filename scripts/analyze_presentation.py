#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import rankdata, wilcoxon

from common import ROOT, manifest_rows, read_rgb

MODELS = ["dncnn", "drunet", "diffusion", "pnp_flow"]
DISPLAY = {"dncnn": "DnCNN", "drunet": "DRUNet", "diffusion": "DiffPIR", "pnp_flow": "PnP-Flow"}
SIGMAS = [15, 25, 50]
BOOTSTRAPS = 10_000


def holm_adjust(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        value = min(1.0, (m-rank)*p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()


def paired_statistics(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(566)
    records = []
    for sigma in SIGMAS:
        for metric in ["psnr", "ssim"]:
            wide = raw[raw.sigma.eq(sigma)].pivot(index="image_id", columns="model", values=metric)
            block = []
            for a, b in itertools.combinations(MODELS, 2):
                diff = (wide[a]-wide[b]).to_numpy()
                samples = rng.integers(0, len(diff), size=(BOOTSTRAPS, len(diff)))
                boot = diff[samples].mean(axis=1)
                try:
                    p = float(wilcoxon(diff, alternative="two-sided", zero_method="wilcox").pvalue)
                except ValueError:
                    p = 1.0
                block.append(dict(
                    sigma=sigma, metric=metric, model_a=a, model_b=b,
                    mean_a=float(wide[a].mean()), mean_b=float(wide[b].mean()),
                    mean_difference_a_minus_b=float(diff.mean()),
                    bootstrap_ci95_low=float(np.quantile(boot, .025)),
                    bootstrap_ci95_high=float(np.quantile(boot, .975)),
                    win_rate_a=float((diff>0).mean()), tie_rate=float((diff==0).mean()),
                    wilcoxon_p=p,
                ))
            adjusted = holm_adjust([r["wilcoxon_p"] for r in block])
            for r, q in zip(block, adjusted):
                r["holm_adjusted_p"] = q
                r["significant_0_05"] = q < .05
                records.append(r)
    paired = pd.DataFrame(records)

    ranks = []
    for sigma in SIGMAS:
        for metric in ["psnr", "ssim"]:
            wide = raw[raw.sigma.eq(sigma)].pivot(index="image_id", columns="model", values=metric)[MODELS]
            rank_matrix = np.vstack([rankdata(-row, method="average") for row in wide.to_numpy()])
            for j, model in enumerate(MODELS):
                ranks.append(dict(sigma=sigma, metric=metric, model=model,
                                  average_rank=float(rank_matrix[:,j].mean()),
                                  first_place_rate=float((rank_matrix[:,j]==1).mean())))
    return paired, pd.DataFrame(ranks)


def select_cases(raw: pd.DataFrame) -> pd.DataFrame:
    used = set(); selected = []
    def choose(case_type, sigma, score, explanation):
        candidates = score.sort_values(ascending=False)
        image_id = next(i for i in candidates.index if i not in used)
        used.add(image_id)
        selected.append(dict(case_type=case_type, image_id=image_id, sigma=sigma,
                             selection_score=float(candidates.loc[image_id]), rationale=explanation))

    # Representative: closest standardized metric vector to its column medians at sigma=25.
    s25 = raw[raw.sigma.eq(25)].pivot(index="image_id", columns="model", values=["psnr","ssim"])
    z = (s25-s25.median())/s25.std(ddof=0)
    choose("representative_sigma25", 25, -np.sqrt((z*z).sum(axis=1)),
           "Closest multivariate PSNR/SSIM profile to the sigma=25 medians across all four models.")
    p50 = raw[raw.sigma.eq(50)].pivot(index="image_id", columns="model", values="ssim")
    choose("pnp_flow_structure_advantage", 50, p50.pnp_flow-p50.drunet,
           "Largest PnP-Flow minus DRUNet SSIM difference at sigma=50.")
    p25 = raw[raw.sigma.eq(25)].pivot(index="image_id", columns="model", values="psnr")
    choose("drunet_over_dncnn", 25, p25.drunet-p25.dncnn,
           "Largest DRUNet minus DnCNN PSNR difference at sigma=25.")
    other_best = p25[["dncnn","drunet","pnp_flow"]].max(axis=1)
    choose("diffpir_domain_limitation", 25, other_best-p25.diffusion,
           "Largest gap between DiffPIR and the best non-diffusion method at sigma=25.")
    return pd.DataFrame(selected)


def best_crop(clean: np.ndarray, a: np.ndarray, b: np.ndarray, size: int=96) -> tuple[int,int]:
    # Find a non-overlapping-grid crop where method A and B differ most in absolute-error advantage.
    ea=np.abs(clean-a).mean(2); eb=np.abs(clean-b).mean(2); score=np.abs(ea-eb)
    best=(-1.0,0,0)
    for y in range(0, clean.shape[0]-size+1, 16):
        for x in range(0, clean.shape[1]-size+1, 16):
            value=float(score[y:y+size,x:x+size].mean())
            if value>best[0]: best=(value,y,x)
    return best[1],best[2]


def make_case_figures(raw: pd.DataFrame, cases: pd.DataFrame) -> None:
    outdir=ROOT/"artifacts/presentation/cases"; outdir.mkdir(parents=True,exist_ok=True)
    manifest={x["image_id"]:x for x in manifest_rows()}
    rows=[]
    for _,case in cases.iterrows():
        image_id=case.image_id; sigma=int(case.sigma); item=manifest[image_id]
        clean=read_rgb(ROOT/item["clean_path"])
        noisy=read_rgb(ROOT/"data/noisy"/f"sigma{sigma}"/f"{image_id}.png")
        outputs={m:read_rgb(ROOT/"outputs"/m/f"sigma{sigma}"/f"{image_id}.png") for m in MODELS}
        if case.case_type=="pnp_flow_structure_advantage": a,b="pnp_flow","drunet"
        elif case.case_type=="drunet_over_dncnn": a,b="drunet","dncnn"
        elif case.case_type=="diffpir_domain_limitation": a,b="diffusion","drunet"
        else: a,b="drunet","pnp_flow"
        y,x=best_crop(clean,outputs[a],outputs[b]); size=96
        images=[clean,noisy]+[outputs[m] for m in MODELS]; labels=["Clean","Noisy"]+[DISPLAY[m] for m in MODELS]
        fig,axes=plt.subplots(2,6,figsize=(17,6.1))
        for col,(im,label) in enumerate(zip(images,labels)):
            axes[0,col].imshow(im); axes[0,col].set_title(label,fontsize=12); axes[0,col].axis("off")
            axes[0,col].add_patch(plt.Rectangle((x,y),size,size,fill=False,edgecolor="yellow",linewidth=1.8))
            axes[1,col].imshow(im[y:y+size,x:x+size]); axes[1,col].axis("off")
            if col>=2:
                model=MODELS[col-2]; r=raw[(raw.image_id==image_id)&(raw.sigma==sigma)&(raw.model==model)].iloc[0]
                axes[1,col].set_title(f"{r.psnr:.2f} dB | {r.ssim:.3f}",fontsize=10)
        fig.suptitle(f"{case.case_type} | {image_id} | sigma={sigma}\nPost-hoc explanatory case; yellow box is the objective comparison crop",fontsize=13)
        fig.tight_layout(); path=outdir/f"{case.case_type}.png"; fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)
        row=case.to_dict(); row.update(crop_x=x,crop_y=y,crop_size=size,figure=path.relative_to(ROOT).as_posix()); rows.append(row)
    pd.DataFrame(rows).to_csv(ROOT/"artifacts/tables/presentation_cases.csv",index=False)


def write_summary(raw: pd.DataFrame, paired: pd.DataFrame, ranks: pd.DataFrame, cases: pd.DataFrame) -> None:
    lines=["# Presentation statistical analysis", "",
           "All inferential comparisons are paired by image. Confidence intervals use 10,000 paired bootstrap resamples with seed 566. Wilcoxon signed-rank p-values are Holm-adjusted across the six model pairs within each `(sigma, metric)` family.", "",
           "## Key paired comparisons", ""]
    targets=[("drunet","dncnn"),("drunet","diffusion"),("drunet","pnp_flow"),("pnp_flow","diffusion")]
    for sigma in SIGMAS:
        lines += [f"### Sigma = {sigma}", "", "| Metric | Comparison | Mean difference | 95% paired-bootstrap CI | A win rate | Holm p |", "|---|---|---:|---:|---:|---:|"]
        for metric in ["psnr","ssim"]:
            for a,b in targets:
                r=paired[(paired.sigma==sigma)&(paired.metric==metric)&(paired.model_a==a)&(paired.model_b==b)]
                sign=1
                if r.empty:
                    r=paired[(paired.sigma==sigma)&(paired.metric==metric)&(paired.model_a==b)&(paired.model_b==a)]; sign=-1
                q=r.iloc[0]; diff=sign*q.mean_difference_a_minus_b; lo,hi=(q.bootstrap_ci95_low,q.bootstrap_ci95_high) if sign==1 else (-q.bootstrap_ci95_high,-q.bootstrap_ci95_low); win=q.win_rate_a if sign==1 else 1-q.win_rate_a-q.tie_rate
                lines.append(f"| {metric.upper()} | {DISPLAY[a]} − {DISPLAY[b]} | {diff:.4f} | [{lo:.4f}, {hi:.4f}] | {win:.1%} | {q.holm_adjusted_p:.3g} |")
        lines.append("")
    lines += ["## Average ranks", "", "Lower rank is better.", "", "| Sigma | Metric | DnCNN | DRUNet | DiffPIR | PnP-Flow |", "|---:|---|---:|---:|---:|---:|"]
    for sigma in SIGMAS:
        for metric in ["psnr","ssim"]:
            q=ranks[(ranks.sigma==sigma)&(ranks.metric==metric)].set_index("model")
            lines.append(f"| {sigma} | {metric.upper()} | "+" | ".join(f"{q.loc[m,'average_rank']:.2f}" for m in MODELS)+" |")
    lines += ["", "## Objectively selected post-hoc cases", "", "These cases explain observed behavior and are separate from the six pre-fixed qualitative examples used in the main experiment.", "", "| Case | Image | Sigma | Selection rule | Figure |", "|---|---|---:|---|---|"]
    for _,c in cases.iterrows():
        lines.append(f"| {c.case_type} | `{c.image_id}` | {int(c.sigma)} | {c.rationale} | `artifacts/presentation/cases/{c.case_type}.png` |")
    lines += ["", "## Files for slides", "", "- `artifacts/tables/paired_comparisons.csv`: every pairwise comparison.", "- `artifacts/tables/model_ranks.csv`: average ranks and first-place rates.", "- `artifacts/tables/presentation_cases.csv`: case selection scores and crop coordinates.", "- `artifacts/presentation/cases/`: slide-ready case figures.", "- `artifacts/plots/`: PSNR, SSIM, and quality/runtime plots."]
    (ROOT/"PRESENTATION_DATA.md").write_text("\n".join(lines)+"\n")


def main() -> None:
    raw=pd.read_csv(ROOT/"results/raw_metrics.csv")
    assert len(raw)==1200 and not raw.duplicated(["image_id","model","sigma"]).any()
    paired,ranks=paired_statistics(raw); cases=select_cases(raw)
    paired.to_csv(ROOT/"artifacts/tables/paired_comparisons.csv",index=False)
    ranks.to_csv(ROOT/"artifacts/tables/model_ranks.csv",index=False)
    make_case_figures(raw,cases); write_summary(raw,paired,ranks,cases)
    print(cases.to_string(index=False)); print(f"Wrote {len(paired)} paired comparisons and {len(ranks)} rank rows")


if __name__=="__main__": main()
