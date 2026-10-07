# STOR 566 图像去噪课程项目：实验设计与结果

## 1. 实验目标

本项目比较四种具有代表性的预训练图像去噪或图像恢复方法：经典残差卷积网络 DnCNN、U-Net 风格去噪网络 DRUNet、基于扩散模型的 DiffPIR，以及基于流匹配先验的 PnP-Flow。

本实验只使用模型作者公开的预训练权重，不训练、微调或更新任何模型参数。所有方法处理完全相同的干净图像、空间分辨率和噪声实现，以保证比较具有可重复性。

主要研究问题是：

> 在 AFHQ-Cat 图像上，经典判别式去噪网络与生成式图像恢复方法在恢复质量、结构保持能力和计算成本之间有怎样的权衡？

## 2. 计算环境

实验运行在以下硬件和软件环境中：

| 项目 | 配置 |
|---|---|
| GPU | 8 × NVIDIA RTX A5000 |
| 单卡显存 | 24,564 MiB，约 24 GB |
| NVIDIA Driver | 570.86.10 |
| 系统报告的 CUDA 上限 | CUDA 12.8 |
| PyTorch CUDA | CUDA 12.1 |
| cuDNN | 9.1.0 |
| Python | 3.10.19 |
| PyTorch | 2.5.1 |
| torchvision | 0.20.1 |
| DeepInv | 0.3.2 |
| scikit-image | 0.25.2 |

每次模型推理的 batch size 均为 1，而且一次推理只使用一张 GPU。为了减少总实验时间，完整数据集被划分成互不重叠的确定性分片，并在多张型号相同的 RTX A5000 上并行运行。多 GPU 只降低整体墙钟时间，不改变单张图像的推理算法、batch size 或显存测量方法。

完整环境记录见：

- `environment/system.txt`
- `environment/pip_freeze.txt`
- `environment/conda_environment.yml`

## 3. 数据集

### 3.1 数据来源

主实验使用 **AFHQ-Cat** 数据集。数据通过 PnP-Flow 官方仓库提供的下载地址获取，并严格按照该仓库的划分规则构造测试集。

原始 AFHQ 数据将 `val` 作为官方 PnP-Flow 代码中的 `test`。随后，PnP-Flow 仓库使用固定的 `validation_images.txt` 清单，将其中 31 张猫图像移动到单独的验证集。完成这一划分后，AFHQ-Cat 测试集中共有 **469 张图像**。

### 3.2 为什么使用 100 张图像

本实验没有声称使用完整的 469 张测试图像，而是按照预先制定的实验方案使用其中一个确定性的 100 张子集：

1. 将 469 张测试图像按相对路径排序；
2. 在任何模型评测或结果检查之前选择排序后的前 100 张；
3. 将最终列表固定在 `artifacts/test_manifest.txt`；
4. 四种方法和三个噪声等级均使用相同的 100 张图像。

100 张的规模来自项目 handoff 中预先规定的实验设计。它为每个“模型—噪声等级”组合提供 100 个配对观测，同时将迭代式 PnP-Flow 的计算量控制在合理范围内。

因此，报告中的正确表述是：

> We evaluated a deterministic 100-image subset of the 469-image AFHQ-Cat test split used by the official PnP-Flow implementation. Images were sorted by relative path, and the first 100 were selected before model evaluation.

### 3.3 图像预处理

- 所有图像均读取为 RGB；
- 空间分辨率统一为 `256 × 256`；
- 共享评估范围为 `[0,1]`；
- 干净图像保存为无损 PNG；
- 不使用 JPEG 格式保存加噪后的图像；
- 不裁剪边界；
- 所有输出在评估前限制到 `[0,1]`。

## 4. 噪声模型

对每张干净图像 \(x\in[0,1]\)，生成加性高斯白噪声：

\[
y=\operatorname{clip}(x+n,0,1),\qquad
n\sim\mathcal{N}\left(0,(\sigma/255)^2\right).
\]

实验使用三个噪声等级：

\[
\sigma\in\{15,25,50\}.
\]

每个 `(image_id, sigma)` 使用唯一且确定性的随机种子。噪声只生成一次，并以 float32 `.npy` 文件保存。四个模型均直接读取相同的 `.npy` 文件，而不是在各自的官方代码中重新随机加噪。因此，相同图像和噪声等级在所有方法间具有逐像素完全相同的输入。

总实验规模为：

\[
100\ \text{images}\times3\ \text{noise levels}\times4\ \text{models}
=1200\ \text{model outputs}.
\]

## 5. 比较方法

### 5.1 DnCNN

DnCNN 是经典的残差卷积去噪网络。网络不是直接预测干净图像，而是预测输入图像中的噪声残差，最终输出为：

\[
\hat{x}=y-f_{\theta}(y).
\]

本实验使用 DnCNN 官方仓库中的 RGB、噪声等级专用权重：

- `color_sigma=15.mat`
- `color_sigma=25.mat`
- `color_sigma=50.mat`

这些权重为 MATLAB v7.3/MatConvNet 格式。实验适配器通过 HDF5 读取官方卷积核和偏置，并将其加载到等价的 PyTorch 卷积层中。该适配只改变权重存储格式和输入输出接口，不改变网络或参数。

### 5.2 DRUNet

DRUNet 是 DPIR 工作中使用的残差 U-Net 风格深度去噪器。本实验使用作者官方发布的 `drunet_color.pth`。

DRUNet 接收四通道输入：前三个通道是 `[0,1]` RGB 噪声图像，第四个通道是常数噪声图：

\[
M_{i,j}=\sigma/255.
\]

同一个预训练 DRUNet color checkpoint 支持 \(\sigma=15,25,50\) 三种噪声强度。

### 5.3 DiffPIR

PnP-Flow 官方代码中的 `pnp_diff` 已通过源代码审计确认是：

> DiffPIR — *Denoising Diffusion Models for Plug-and-Play Image Restoration*，Zhu et al., CVPRW/NTIRE 2023。

它不是 DDRM，因此本报告不使用 DDRM 这一名称。

官方代码通过 DeepInv 构造 `DiffUNet()`，并使用 `deepinv.sampling.DiffPIR` 进行图像恢复。实验参数为：

| 参数 | 值 |
|---|---:|
| 扩散迭代数 | 100 |
| `lambda` | 1.0 |
| `zeta` | 1.0 |
| Data fidelity | L2 |
| Observation operator | Identity/denoising |

输入范围为 `[0,1]`，观测噪声标准差传入为 `sigma/255`。

需要注意，DeepInv 自动提供的官方 checkpoint 是 `diffusion_ffhq_10m.pt`，训练域为 FFHQ 人脸，而测试集为 AFHQ-Cat 猫图像。这是实验的重要公平性限制。

### 5.4 PnP-Flow

PnP-Flow 的完整名称是：

> *PnP-Flow: Plug-and-Play Image Restoration with Flow Matching*，Gagneux, Martin and Hagemann, ICLR 2025。

本实验使用 PnP-Flow 作者发布的 AFHQ-Cat optimal-transport flow-matching U-Net checkpoint。推理参数来自官方 denoising 配置：

| 参数 | 值 |
|---|---:|
| PnP steps | 100 |
| 每一步 Monte Carlo samples | 5 |
| 总模型调用次数 NFE | 500 |
| `lr_pnp` | 1.0 |
| `alpha` | 0.8 |
| 学习率策略 | `alpha_1_minus_t` |

PnP-Flow 在 `[-1,1]` 范围内运行。共享的 `[0,1]` 噪声输入通过确定性线性变换映射至 `[-1,1]`，因此对应的噪声标准差为 `2*sigma/255`。输出随后映射回 `[0,1]` 进行公共评估。

## 6. 评价指标

### 6.1 PSNR

对 `[0,1]` RGB 图像计算：

\[
\operatorname{PSNR}(x,\hat{x})
=10\log_{10}\left(\frac{1}{\operatorname{MSE}(x,\hat{x})}\right).
\]

PSNR 越高表示平均像素误差越小，单位为 dB。

### 6.2 SSIM

使用 scikit-image 0.25.2 的 `structural_similarity`，所有方法采用相同设置：

- `data_range=1.0`
- `channel_axis=2`
- `gaussian_weights=True`
- Gaussian sigma = 1.5
- `use_sample_covariance=False`

SSIM 越高表示结构和局部对比度保持越好。

### 6.3 运行效率

对每张图像记录：

- 推理时间，单位为秒/图；
- PyTorch peak allocated GPU memory，单位为 MiB；
- DiffPIR 和 PnP-Flow 的模型调用或迭代次数。

计时不包括模型加载、checkpoint 加载、数据读取和输出文件写入。每种模型在正式计时前执行至少两次 warm-up，并在开始和结束计时前调用 `torch.cuda.synchronize()`。

## 7. 实验结果

### 7.1 主要质量结果

表中数值为 100 张图像的 mean ± standard deviation。

| 方法 | σ | PSNR (dB) | SSIM |
|---|---:|---:|---:|
| Noisy input | 15 | 24.800 ± 0.207 | 0.5122 ± 0.0792 |
| DnCNN | 15 | 35.261 ± 1.370 | 0.9321 ± 0.0163 |
| DRUNet | 15 | **35.590 ± 1.425** | **0.9368 ± 0.0160** |
| DiffPIR | 15 | 31.638 ± 0.687 | 0.8201 ± 0.0310 |
| PnP-Flow | 15 | 31.903 ± 1.637 | 0.8621 ± 0.0393 |
| Noisy input | 25 | 20.536 ± 0.291 | 0.3232 ± 0.0728 |
| DnCNN | 25 | 32.461 ± 1.466 | 0.8883 ± 0.0268 |
| DRUNet | 25 | **32.838 ± 1.541** | **0.8968 ± 0.0263** |
| DiffPIR | 25 | 30.044 ± 1.073 | 0.7904 ± 0.0342 |
| PnP-Flow | 25 | 31.271 ± 1.623 | 0.8558 ± 0.0394 |
| Noisy input | 50 | 15.037 ± 0.383 | 0.1413 ± 0.0433 |
| DnCNN | 50 | 27.817 ± 1.597 | 0.7875 ± 0.0502 |
| DRUNet | 50 | **28.179 ± 1.677** | 0.8027 ± 0.0491 |
| DiffPIR | 50 | 26.805 ± 1.366 | 0.7272 ± 0.0469 |
| PnP-Flow | 50 | 28.143 ± 1.692 | **0.8094 ± 0.0431** |

### 7.2 效率结果

| 方法 | 平均时间/图 | 最大峰值显存 | NFE/迭代 |
|---|---:|---:|---:|
| DnCNN | 0.0062 s | 37.2 MiB | 单次前向传播 |
| DRUNet | 0.0114 s | 222.7 MiB | 单次前向传播 |
| DiffPIR | 约 2.98 s | 664.4 MiB | 100 |
| PnP-Flow | 约 11.24 s | 293.3 MiB | 500 |

### 7.3 结果分析

1. **DRUNet 在低噪声和中等噪声下表现最好。** 在 \(\sigma=15\) 和 \(\sigma=25\) 时，DRUNet 同时取得最高平均 PSNR 和 SSIM。

2. **DnCNN 的性价比很高。** 虽然质量略低于 DRUNet，但每张图像只需要约 0.0062 秒，且峰值显存不到 40 MiB。

3. **PnP-Flow 在高噪声下具有较强的结构保持能力。** 在 \(\sigma=50\) 时，其平均 SSIM 为 0.8094，是四种模型中最高的；PSNR 28.143 dB 与 DRUNet 的 28.179 dB 非常接近。

4. **PnP-Flow 的计算成本明显更高。** 由于每张图像需要 100 个 PnP step，且每步计算 5 个 flow samples，其推理时间约为 11.2 秒/图。

5. **DiffPIR 在本实验中表现较弱，不能简单归因于扩散模型本身。** 它使用的是 FFHQ 人脸先验，而测试图像是猫。结果更准确地反映了“官方 FFHQ DiffPIR prior 在 AFHQ-Cat 上的跨域表现”，而不是一个域匹配扩散模型的理论上限。

## 8. 可靠性检查

最终自动检查结果为：

- 100 张 manifest 图像无重复；
- 300 个 `(image, sigma)` 噪声张量均唯一；
- 1200 个预期模型输出全部存在；
- 1200 行 raw metrics 键全部唯一；
- 所有输出尺寸均为 `256 × 256 × 3`；
- 所有输入和输出均为有限数值；
- 所有评估值均处于 `[0,1]`；
- 随机抽取 5%，即 60 行结果重新计算 PSNR/SSIM，与 CSV 数值一致；
- 没有失败、NaN、GPU OOM 或缺失输出。

## 9. 局限性

1. 主实验只使用 469 张官方测试图像中的固定 100 张，因此结果不能表述为“完整 AFHQ-Cat test split”的结果。
2. DiffPIR 使用 FFHQ checkpoint，而其他生成式先验中 PnP-Flow 使用 AFHQ-Cat checkpoint，先验训练域并不完全公平。
3. 本实验比较的是作者推荐或官方代码提供的固定设置，没有进行大规模超参数搜索。
4. 当前表格给出均值和标准差；若课程强调统计推断，可进一步加入逐图配对 bootstrap 置信区间和模型间胜率。
5. DnCNN checkpoint 经由确定性的 MatConvNet-to-PyTorch I/O 适配读取；正式论文中应明确披露这一实现细节。

## 10. 可复现性

官方仓库固定版本：

| 仓库 | Commit |
|---|---|
| DnCNN | `e93b27812d3ff523a3a79d19e5e50d233d7a8d0a` |
| DPIR | `15bca3fcc1f3cc51a1f99ccf027691e278c19354` |
| PnP-Flow | `ee5b159e6edff4d585ebf36c9a5333ffa85e066c` |

主要 checkpoint SHA-256：

| Checkpoint | SHA-256 |
|---|---|
| DnCNN σ=15 | `c97c63c7cdee39f118c743fb0ada5c54c7074aa7aefa892e667f4f0440b8d893` |
| DnCNN σ=25 | `99ac1aaaa4f851a48e5a686d13de05cf8dd8863606b477d1a917e8004e9e42cf` |
| DnCNN σ=50 | `84632c872b1191826f1c493eba6926cf626173b80c2b52889401616ee1eb7ad5` |
| DRUNet color | `479abe3c5327dfd10ff54a80ec7d4098ca80752a5c9492cdff31cee430bec4b4` |
| DiffPIR DiffUNet | `81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa` |
| PnP-Flow AFHQ-Cat OT | `c8f1b67c623f9bd4b62ff588b2480873fb5797a75118b2803657fdf134e689af` |

顺序复现实验的命令为：

```bash
cd /data2/yangyz/stor566-denoising

environment/conda/bin/python scripts/prepare_data.py
CUDA_VISIBLE_DEVICES=0 environment/conda/bin/python scripts/run_dncnn.py
CUDA_VISIBLE_DEVICES=0 environment/conda/bin/python scripts/run_drunet.py
CUDA_VISIBLE_DEVICES=0 environment/conda/bin/python scripts/run_diffusion.py
CUDA_VISIBLE_DEVICES=0 environment/conda/bin/python scripts/run_pnp_flow.py

environment/conda/bin/python scripts/sanity_check.py
environment/conda/bin/python scripts/summarize_results.py
```

原始结果和图表位置：

- 逐图指标：`results/raw_metrics.csv`
- 汇总指标：`results/summary.csv`
- 主质量表：`artifacts/tables/main_quality.csv`
- 效率表：`artifacts/tables/efficiency.csv`
- PSNR 曲线：`artifacts/plots/psnr_vs_sigma.png`
- SSIM 曲线：`artifacts/plots/ssim_vs_sigma.png`
- 质量—时间曲线：`artifacts/plots/quality_runtime_tradeoff.png`
- 定性结果和误差图：`artifacts/qualitative/`

核心实验 wrapper、配置及结果对应的本地 Git commit 为：

```text
30b94d8b06978e5b0fe0ec29f3f5cf2e4eeeab07
```

## 11. 总结

在 AFHQ-Cat 固定 100 张测试子集上，DRUNet 提供了最稳定的整体像素恢复质量，同时保持非常低的推理成本。DnCNN 的质量略低，但速度最快且显存占用最低。PnP-Flow 在高噪声条件下取得最高 SSIM，说明域匹配的生成式流先验有助于保持图像结构，但其推理成本远高于前馈 CNN。DiffPIR 受到 FFHQ-to-AFHQ 跨域先验不匹配的明显影响，因此其结果应结合 checkpoint 域差异谨慎解释。
