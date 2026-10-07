# Experiment run log

## Compatibility audit

Audit date: 2026-10-07 (America/Los_Angeles). No model was trained or fine-tuned.

### Dataset — READY with documented preprocessing adapter

- Official PnP-Flow commit `ee5b159e6edff4d585ebf36c9a5333ffa85e066c` expects AFHQ-Cat at 256×256 (`config/dataset_config/afhq_cat.yaml`).
- The official `download.sh` obtains the AFHQ archive and flow checkpoint separately. The raw AFHQ `val` split is renamed to PnP-Flow `test`; `scripts/afhq_validation_images.sh` then moves a fixed 31-image list into a new validation split. This leaves 469 test cat images.
- `pnpflow/dataloaders.py` resizes to 256×256, converts RGB to a tensor, and normalizes `[0,1]` to `[-1,1]`.
- We sort the resulting test paths and freeze the first 100 in `artifacts/test_manifest.txt`. The processed clean images are saved once as RGB PNG. Each clipped noisy tensor is generated once in `[0,1]` float32 and saved as `.npy`; every method reads that exact file.
- The source loader's internal noise generation is deliberately bypassed. This is an I/O adapter only.

### DnCNN — NEEDS_DOCUMENTED_ADAPTER (implemented)

- Official RGB sigma-specific checkpoints are `model/specifics_color/color_sigma={15,25,50}.mat` in the DnCNN repository.
- `Demo_test_CDnCNN_Specific.m` confirms the network predicts residual noise and the denoised result is `input - residual`.
- The checkpoints contain 20 convolution layers with intervening ReLU layers, RGB input/output, and no normalization transform beyond `[0,1]` image values.
- `scripts/run_dncnn.py` reads the official MATLAB v7.3/MatConvNet tensors through HDF5 and installs them unchanged into equivalent PyTorch convolutions. This changes checkpoint representation/I/O only.

### DRUNet — READY with documented I/O adapter

- Official checkpoint `drunet_color.pth` is downloaded from the author-controlled KAIR GitHub release referenced by DPIR's `main_download_pretrained_models.py`.
- `main_dpir_denoising.py` constructs `UNetRes(in_nc=4,out_nc=3,...)`, expects RGB `[0,1]`, and concatenates a constant `sigma/255` fourth channel.
- A single color checkpoint supports all three noise levels. `scripts/run_drunet.py` follows this interface exactly.

### Diffusion — READY with documented I/O adapter and fairness limitation

- PnP-Flow's `pnpflow/methods/pnp_diff.py` instantiates `deepinv.sampling.DiffPIR`; the cited method is **Denoising Diffusion Models for Plug-and-Play Image Restoration** (Zhu et al., CVPRW/NTIRE 2023). It is not DDRM.
- `pnpflow/utils.py` constructs `deepinv.models.DiffUNet()`. DeepInv 0.3.2 resolves this to the official `diffusion_ffhq_10m.pt` checkpoint, a 256×256 FFHQ prior.
- DiffPIR uses an identity denoising physics operator, L2 data fidelity, 100 iterations, `lambda=1.0`, and `zeta=1.0`, matching the official test script's denoising settings.
- It accepts Gaussian sigma in `[0,1]`; `scripts/run_diffusion.py` passes `sigma/255` and the exact saved noisy tensor.
- Fairness limitation: the official DiffPIR checkpoint is FFHQ-domain, while evaluation is AFHQ-Cat. This is the checkpoint actually invoked by official PnP-Flow code, but its domain mismatch must be disclosed.

### Flow — READY with documented I/O/range adapter

- Method: **PnP-Flow: Plug-and-Play Image Restoration with Flow Matching** (Gagneux, Martin, Hagemann; ICLR 2025).
- Official AFHQ-Cat OT flow-matching U-Net checkpoint is loaded from the README's Google Drive ID.
- Author-recommended denoising configuration from `scripts/script_test.sh`: 100 PnP steps, 5 Monte Carlo samples per step, `lr_pnp=1.0`, `alpha=0.8`, and `gamma_style=alpha_1_minus_t` (500 model evaluations/image).
- The flow operates in `[-1,1]`. Exact saved `[0,1]` noisy tensors are linearly mapped to `[-1,1]`, so the likelihood noise standard deviation is correspondingly `2*sigma/255`. Output is mapped back and clamped for shared evaluation.
- Ground truth is never used in inference or per-image tuning. No test-set parameter tuning was performed.

## Execution gates

- Gate 1: all checkpoints load strictly; one 256×256 RGB tensor passes through each model; outputs are finite and correctly shaped.
- Gate 2: smoke outputs preserve identity/geometry and metrics/timing/memory are recorded. Iterative smoke runtimes were approximately 3.0 s/image for DiffPIR and 11.3 s/image for PnP-Flow on RTX A5000.
- Gate 3: ten-image behavior was inspected during the incremental run; all branches use the same `.npy` noisy inputs, RGB order, and resolution. No NaNs, identical-image failures, or OOMs occurred.
- Gate 4: completed on 100 images × 3 sigma levels × 4 models. See final summary and sanity-check output.

## Deviations and hardware

The handoff anticipated one NVIDIA A40. The available machine instead has eight NVIDIA RTX A5000 GPUs with 24,564 MiB each. Batch size remained 1; independent deterministic image shards were distributed across GPUs only to reduce wall-clock time.

