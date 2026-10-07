#!/usr/bin/env python3
from __future__ import annotations
import json
import random
import pandas as pd
import numpy as np
from common import ROOT, manifest_rows, metrics, read_rgb

def main():
    manifest=manifest_rows(); assert len(manifest)==100; assert len({x['relative_path'] for x in manifest})==100
    raw=pd.read_csv(ROOT/"results/raw_metrics.csv"); expected=100*3*4
    assert len(raw)==expected, (len(raw),expected); assert not raw.duplicated(["image_id","sigma","model"]).any(); assert (raw.status=="ok").all()
    for item in manifest:
      for sigma in (15,25,50):
        noisy=np.load(ROOT/"data/noisy"/f"sigma{sigma}"/f"{item['image_id']}.npy",allow_pickle=False); assert noisy.shape==(256,256,3) and np.isfinite(noisy).all() and noisy.min()>=0 and noisy.max()<=1
    sample=raw.sample(n=max(1,len(raw)//20),random_state=566)
    index={x['image_id']:x for x in manifest}
    for _,r in sample.iterrows():
        clean=read_rgb(ROOT/index[r.image_id]['clean_path']); out=read_rgb(ROOT/r.output_path); assert out.shape==clean.shape and np.isfinite(out).all(); p,s=metrics(clean,out); assert abs(p-r.psnr)<1e-6 and abs(s-r.ssim)<1e-6
    print(f"PASS: {len(raw)} unique model outputs; recomputed {len(sample)} metric rows")
if __name__=="__main__": main()
