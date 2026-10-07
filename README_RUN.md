# STOR 566 pretrained denoising benchmark

This project evaluates official pretrained DnCNN, DRUNet, DiffPIR, and PnP-Flow models on the same 100 AFHQ-Cat test images and the same saved Gaussian-noise tensors. No training or fine-tuning is performed.

## Environment

```bash
cd /data2/yangyz/stor566-denoising
conda env create -p environment/conda -f environment/conda_environment.yml
```

The completed local environment can be used directly as `environment/conda/bin/python`.

## Reproduce

```bash
environment/conda/bin/python scripts/prepare_data.py
CUDA_VISIBLE_DEVICES=0 environment/conda/bin/python scripts/run_dncnn.py
CUDA_VISIBLE_DEVICES=0 environment/conda/bin/python scripts/run_drunet.py
CUDA_VISIBLE_DEVICES=0 environment/conda/bin/python scripts/run_diffusion.py
CUDA_VISIBLE_DEVICES=0 environment/conda/bin/python scripts/run_pnp_flow.py
environment/conda/bin/python scripts/sanity_check.py
environment/conda/bin/python scripts/summarize_results.py
```

All runners are resumable: a completed `(image, model, sigma)` row is skipped. The iterative runners also accept `--start` and `--limit` for non-overlapping GPU shards.

The external repositories and checkpoints are intentionally ignored by the wrapper Git repository. Their exact sources, commits, and SHA-256 values are recorded under `environment/`.

