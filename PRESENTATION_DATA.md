# Presentation statistical analysis

The instructor-feedback experiments, including CBSD68, Laplace corruption, PSNR/SSIM formulas, new statistical tests and slide-ready cases, are documented in [`SUPPLEMENTARY_RESULTS.md`](SUPPLEMENTARY_RESULTS.md).

All inferential comparisons are paired by image. Confidence intervals use 10,000 paired bootstrap resamples with seed 566. Wilcoxon signed-rank p-values are Holm-adjusted across the six model pairs within each `(sigma, metric)` family.

## Key paired comparisons

### Sigma = 15

| Metric | Comparison | Mean difference | 95% paired-bootstrap CI | A win rate | Holm p |
|---|---|---:|---:|---:|---:|
| PSNR | DRUNet − DnCNN | 0.3292 | [0.3136, 0.3451] | 100.0% | 2.34e-17 |
| PSNR | DRUNet − DiffPIR | 3.9519 | [3.8020, 4.1083] | 100.0% | 2.34e-17 |
| PSNR | DRUNet − PnP-Flow | 3.6870 | [3.6063, 3.7678] | 100.0% | 2.34e-17 |
| PSNR | PnP-Flow − DiffPIR | 0.2649 | [0.0670, 0.4640] | 59.0% | 0.0378 |
| SSIM | DRUNet − DnCNN | 0.0047 | [0.0045, 0.0049] | 100.0% | 2.34e-17 |
| SSIM | DRUNet − DiffPIR | 0.1166 | [0.1108, 0.1223] | 100.0% | 2.34e-17 |
| SSIM | DRUNet − PnP-Flow | 0.0746 | [0.0694, 0.0802] | 100.0% | 2.34e-17 |
| SSIM | PnP-Flow − DiffPIR | 0.0420 | [0.0321, 0.0519] | 78.0% | 4.16e-11 |

### Sigma = 25

| Metric | Comparison | Mean difference | 95% paired-bootstrap CI | A win rate | Holm p |
|---|---|---:|---:|---:|---:|
| PSNR | DRUNet − DnCNN | 0.3768 | [0.3563, 0.3983] | 100.0% | 2.34e-17 |
| PSNR | DRUNet − DiffPIR | 2.7946 | [2.6800, 2.9149] | 100.0% | 2.34e-17 |
| PSNR | DRUNet − PnP-Flow | 1.5668 | [1.5091, 1.6247] | 100.0% | 2.34e-17 |
| PSNR | PnP-Flow − DiffPIR | 1.2278 | [1.0910, 1.3739] | 99.0% | 2.34e-17 |
| SSIM | DRUNet − DnCNN | 0.0085 | [0.0081, 0.0089] | 100.0% | 2.34e-17 |
| SSIM | DRUNet − DiffPIR | 0.1064 | [0.1007, 0.1123] | 100.0% | 2.34e-17 |
| SSIM | DRUNet − PnP-Flow | 0.0410 | [0.0377, 0.0445] | 100.0% | 2.34e-17 |
| SSIM | PnP-Flow − DiffPIR | 0.0654 | [0.0575, 0.0735] | 98.0% | 2.34e-17 |

### Sigma = 50

| Metric | Comparison | Mean difference | 95% paired-bootstrap CI | A win rate | Holm p |
|---|---|---:|---:|---:|---:|
| PSNR | DRUNet − DnCNN | 0.3620 | [0.3368, 0.3877] | 100.0% | 2.34e-17 |
| PSNR | DRUNet − DiffPIR | 1.3741 | [1.2838, 1.4692] | 100.0% | 2.34e-17 |
| PSNR | DRUNet − PnP-Flow | 0.0362 | [-0.0140, 0.0889] | 51.0% | 0.331 |
| PSNR | PnP-Flow − DiffPIR | 1.3379 | [1.2361, 1.4426] | 99.0% | 2.34e-17 |
| SSIM | DRUNet − DnCNN | 0.0152 | [0.0142, 0.0162] | 100.0% | 2.34e-17 |
| SSIM | DRUNet − DiffPIR | 0.0755 | [0.0681, 0.0832] | 100.0% | 2.34e-17 |
| SSIM | DRUNet − PnP-Flow | -0.0068 | [-0.0089, -0.0046] | 22.0% | 1.24e-08 |
| SSIM | PnP-Flow − DiffPIR | 0.0822 | [0.0757, 0.0890] | 100.0% | 2.34e-17 |

## Average ranks

Lower rank is better.

| Sigma | Metric | DnCNN | DRUNet | DiffPIR | PnP-Flow |
|---:|---|---:|---:|---:|---:|
| 15 | PSNR | 2.00 | 1.00 | 3.59 | 3.41 |
| 15 | SSIM | 2.00 | 1.00 | 3.78 | 3.22 |
| 25 | PSNR | 2.00 | 1.00 | 3.99 | 3.01 |
| 25 | SSIM | 2.01 | 1.00 | 3.98 | 3.01 |
| 50 | PSNR | 2.90 | 1.49 | 3.99 | 1.62 |
| 50 | SSIM | 2.97 | 1.78 | 3.99 | 1.26 |

## Objectively selected post-hoc cases

These cases explain observed behavior and are separate from the six pre-fixed qualitative examples used in the main experiment.

| Case | Image | Sigma | Selection rule | Figure |
|---|---|---:|---|---|
| representative_sigma25 | `058_flickr_cat_000575` | 25 | Closest multivariate PSNR/SSIM profile to the sigma=25 medians across all four models. | `artifacts/presentation/cases/representative_sigma25.png` |
| pnp_flow_structure_advantage | `048_flickr_cat_000514` | 50 | Largest PnP-Flow minus DRUNet SSIM difference at sigma=50. | `artifacts/presentation/cases/pnp_flow_structure_advantage.png` |
| drunet_over_dncnn | `039_flickr_cat_000372` | 25 | Largest DRUNet minus DnCNN PSNR difference at sigma=25. | `artifacts/presentation/cases/drunet_over_dncnn.png` |
| diffpir_domain_limitation | `002_flickr_cat_000016` | 25 | Largest gap between DiffPIR and the best non-diffusion method at sigma=25. | `artifacts/presentation/cases/diffpir_domain_limitation.png` |

## Files for slides

- `artifacts/tables/paired_comparisons.csv`: every pairwise comparison.
- `artifacts/tables/model_ranks.csv`: average ranks and first-place rates.
- `artifacts/tables/presentation_cases.csv`: case selection scores and crop coordinates.
- `artifacts/presentation/cases/`: slide-ready case figures.
- `artifacts/plots/`: PSNR, SSIM, and quality/runtime plots.
