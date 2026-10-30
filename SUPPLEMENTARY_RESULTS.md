# Teacher-feedback experiments: second dataset and non-Gaussian noise

This document records the two experiments added after instructor feedback. They use the same four public checkpoints and the same RTX A5000 hardware as the original experiment.

## Design

| Experiment | Dataset | Images | Corruption | Levels | Purpose |
|---|---|---:|---|---|---|
| Cross-dataset | CBSD68 | 68 | Gaussian | 15, 25, 50 | Test generalization from animal/face priors to natural scenes |
| Noise robustness | AFHQ-Cat | 100 fixed images | Laplace | equivalent SD 15, 25, 50 | Test robustness to heavy-tailed corruption |

CBSD68 was obtained from the model author's official FFDNet repository. To meet the fixed-size requirement of the AFHQ-trained PnP-Flow checkpoint, every image was converted to RGB and bicubic-resized to 256×256. Thus this is a controlled cross-domain comparison, not a claim of native-resolution CBSD68 benchmark numbers.

For Laplace noise, the scale is chosen to match Gaussian standard deviation:

\[
b=\frac{\sigma/255}{\sqrt{2}},\qquad
n\sim\operatorname{Laplace}(0,b),\qquad
y=\operatorname{clip}(x+n,0,1).
\]

All methods receive the exact same saved float32 noisy tensor for each image and level. DnCNN and DRUNet retain their Gaussian-trained checkpoints, so their Laplace results are explicitly zero-shot robustness results. DiffPIR uses an L1 data-fidelity term, and PnP-Flow uses the official Laplace score form. No model is retrained or tuned.

Total supplementary outputs:

\[
(100+68)\times3\times4=2016.
\]

## Metrics and formulas for the presentation

For an RGB image with \(N\) scalar values and range \([0,1]\):

\[
\operatorname{MSE}(x,\hat x)=\frac{1}{N}\sum_{i=1}^{N}(x_i-\hat x_i)^2,
\qquad
\operatorname{PSNR}(x,\hat x)=10\log_{10}\!\left(\frac{1}{\operatorname{MSE}(x,\hat x)}\right).
\]

The local structural-similarity index is

\[
\operatorname{SSIM}(x,\hat x)=
\frac{(2\mu_x\mu_{\hat x}+C_1)(2\sigma_{x\hat x}+C_2)}
{(\mu_x^2+\mu_{\hat x}^2+C_1)(\sigma_x^2+\sigma_{\hat x}^2+C_2)}.
\]

We report the mean of local SSIM values using Gaussian weights (window sigma 1.5), `data_range=1`, and population covariance. Larger PSNR and SSIM are better.

## Results

Values are mean ± sample SD across images.

### AFHQ-Cat with Laplace noise (n=100)

| σeq | Method | PSNR (dB) | SSIM |
|---:|---|---:|---:|
| 15 | DnCNN | 35.227 ± 1.345 | 0.9312 ± 0.0163 |
| 15 | DRUNet | **35.583 ± 1.405** | **0.9362 ± 0.0161** |
| 15 | DiffPIR-L1 | 24.858 ± 0.242 | 0.5230 ± 0.0774 |
| 15 | PnP-Flow-Laplace | 28.148 ± 0.238 | 0.6640 ± 0.0638 |
| 25 | DnCNN | 32.491 ± 1.409 | 0.8871 ± 0.0269 |
| 25 | DRUNet | **32.910 ± 1.496** | **0.8965 ± 0.0267** |
| 25 | DiffPIR-L1 | 20.704 ± 0.316 | 0.3382 ± 0.0729 |
| 25 | PnP-Flow-Laplace | 24.646 ± 0.278 | 0.4993 ± 0.0744 |
| 50 | DnCNN | 28.097 ± 1.491 | 0.7874 ± 0.0495 |
| 50 | DRUNet | **28.466 ± 1.563** | **0.8030 ± 0.0481** |
| 50 | DiffPIR-L1 | 16.299 ± 0.371 | 0.1777 ± 0.0500 |
| 50 | PnP-Flow-Laplace | 20.513 ± 0.369 | 0.3103 ± 0.0659 |

### CBSD68 with Gaussian noise (n=68)

| σ | Method | PSNR (dB) | SSIM |
|---:|---|---:|---:|
| 15 | DnCNN | 33.542 ± 1.865 | 0.9273 ± 0.0237 |
| 15 | DRUNet | **33.839 ± 1.962** | **0.9314 ± 0.0234** |
| 15 | DiffPIR | 30.589 ± 1.040 | 0.8316 ± 0.0479 |
| 15 | PnP-Flow | 28.602 ± 2.239 | 0.7935 ± 0.0713 |
| 25 | DnCNN | 30.595 ± 1.996 | 0.8757 ± 0.0374 |
| 25 | DRUNet | **30.902 ± 2.100** | **0.8823 ± 0.0368** |
| 25 | DiffPIR | 28.531 ± 1.483 | 0.7861 ± 0.0469 |
| 25 | PnP-Flow | 28.128 ± 2.133 | 0.7881 ± 0.0682 |
| 50 | DnCNN | 26.283 ± 2.149 | 0.7517 ± 0.0667 |
| 50 | DRUNet | **26.590 ± 2.252** | **0.7640 ± 0.0673** |
| 50 | DiffPIR | 25.339 ± 1.915 | 0.7014 ± 0.0514 |
| 50 | PnP-Flow | 25.966 ± 2.072 | 0.7463 ± 0.0600 |

## Statistical analysis

Inference is paired by image. For every dataset, noise level and metric, all six model pairs are tested with a two-sided Wilcoxon signed-rank test; the six p-values are Holm-corrected. Group means also have deterministic 10,000-resample bootstrap 95% confidence intervals in `results/supplementary_summary.csv`.

- DRUNet exceeds DnCNN by 0.355, 0.420 and 0.369 dB on Laplace AFHQ-Cat, and by 0.298, 0.308 and 0.306 dB on Gaussian CBSD68. Every PSNR and SSIM comparison has Holm-adjusted p < 1e-11.
- On CBSD68 at σ=50, DRUNet exceeds PnP-Flow by 0.623 dB and 0.0177 SSIM (Holm p < 2.4e-10).
- The strong Laplace result of the two discriminative Gaussian denoisers is genuine zero-shot robustness. The weak DiffPIR-L1 and PnP-Flow-Laplace results show that changing likelihood alone does not overcome prior/domain and hyperparameter mismatch.

The results support a more defensible conclusion than “one model wins universally”: domain-matched discriminative denoisers are consistently strongest here, while the generative plug-and-play methods are sensitive to both prior domain and corruption-specific inference settings.

## Presentation cases

`figures/supplementary_cases.png` shows a deterministic median-difficulty case at σ=50 for each new experiment: clean, noisy, and all four outputs with PSNR/SSIM. Use it to discuss texture preservation, residual heavy-tailed impulses, oversmoothing, and prior mismatch. `figures/supplementary_comparison.png` gives the quality and runtime curves.

## Reproduction and files

```bash
environment/conda/bin/python scripts/prepare_supplementary.py
environment/conda/bin/python scripts/run_supplementary.py --dataset cbsd68 --noise gaussian --model drunet --device cuda:0
environment/conda/bin/python scripts/run_supplementary.py --dataset afhq_cat --noise laplace --model drunet --device cuda:0
environment/conda/bin/python scripts/analyze_supplementary.py
```

- Configuration: `configs/supplementary.yaml`
- Fixed manifests: `artifacts/supplementary/`
- Per-image metrics: `results/supplementary_raw_metrics.csv`
- Means, SDs and bootstrap CIs: `results/supplementary_summary.csv`
- Paired tests: `results/supplementary_paired_tests.csv`
- Large datasets, checkpoints and reconstructed images remain Git-ignored.
