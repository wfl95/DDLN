# DDLN-Noise-Robust Face Recognition via Non-target Similarity Distribution Guided Sample Selection

A lightweight PyTorch implementation of a plug-and-play noise filtering module for noise-robust face recognition training.

This repository provides the core implementation of **DDLN NoiseFilter**, which performs sample filtering directly in cosine-similarity space and can be integrated into common margin-based face recognition losses such as **CosFace** and **ArcFace** with minimal code changes.

> Note: This repository currently provides the core noise filtering module and example margin heads. It is not a complete face recognition training framework.

---

## Overview

Label noise is a common problem in large-scale supervised face recognition datasets. When face images are collected or annotated automatically, incorrect identity labels may be introduced and can mislead model training.

This repository provides a simple and lightweight noise filtering module. The module estimates noise-related similarity statistics during training and progressively filters unreliable samples according to a dynamically updated threshold.

The core idea is to perform noise detection in the cosine-similarity space. During training, the module tracks the lower and upper bounds of noise-related similarity statistics with exponential moving average (EMA), then constructs a progressive filtering threshold from early to later epochs.

---

## Main Features

- PyTorch implementation
- Plug-and-play design
- Compatible with CosFace and ArcFace
- Noise filtering in cosine-similarity space
- Progressive threshold update during training
- EMA-based threshold smoothing
- Minimal modification to existing face recognition training code

---

## Repository Structure

```text
DDLN-NoiseRobustFaceRecognition/
├── README.md
├── noise_filter_DDLN.py
├── requirements.txt
└── .gitignore
```

The main file is:

```text
noise_filter_DDLN.py
```

It contains:

- `NoiseFilter`: the core noise filtering module
- `CosFace`: CosFace margin head with optional denoising
- `ArcFace`: ArcFace margin head with optional denoising
- `build_head`: helper function for constructing CosFace or ArcFace heads

---

## Requirements

The code is implemented with PyTorch.

```text
torch>=1.10
```

You can install the dependency with:

```bash
pip install -r requirements.txt
```

If you do not use `requirements.txt`, install PyTorch manually according to your CUDA environment.

---

## Quick Start

### 1. Import the module

```python
from noise_filter_DDLN import NoiseFilter
```

### 2. Create the noise filter

```python
noise_filter = NoiseFilter(
    log=None,
    milestone0=11,
    gap_epoch=2,
    top_class=10,
    ema_t=0.01,
    alpha=0.02
)
```

### 3. Apply noise filtering before loss computation

```python
cosine_new, label_new = noise_filter(
    cosine,
    label,
    current_epoch
)
```

Then use the filtered cosine similarities and labels to compute the margin-based classification loss.

---

## Example Usage

The following example shows how to use `NoiseFilter` independently.

```python
import torch
from noise_filter_DDLN import NoiseFilter

# Batch size and number of identities/classes
B = 8
C = 1000

# Simulated cosine similarity matrix and labels
cosine = torch.randn(B, C)
label = torch.randint(0, C, (B,))

# Create the noise filter
noise_filter = NoiseFilter(
    log=None,
    milestone0=11,
    gap_epoch=2,
    top_class=10,
    ema_t=0.01,
    alpha=0.02
)

current_epoch = 1

# Filter unreliable samples
cosine_new, label_new = noise_filter(
    cosine,
    label,
    current_epoch
)

print("Before filtering:", cosine.shape, label.shape)
print("After filtering:", cosine_new.shape, label_new.shape)
```

---

## Usage with CosFace

The repository provides a CosFace head with optional noise filtering.

```python
from noise_filter_DDLN import build_head

head = build_head(
    head_type="cosface_denoise",
    m=0.35,
    s=64.0
)

logits, labels = head(
    cosine,
    label,
    current_epoch
)
```

If you do not want to enable denoising, use:

```python
head = build_head(
    head_type="cosface",
    m=0.35,
    s=64.0
)
```

---

## Usage with ArcFace

The repository also provides an ArcFace head with optional noise filtering.

```python
from noise_filter_DDLN import build_head

head = build_head(
    head_type="arcface_denoise",
    m=0.5,
    s=64.0
)

logits, labels = head(
    cosine,
    label,
    current_epoch
)
```

If you do not want to enable denoising, use:

```python
head = build_head(
    head_type="arcface",
    m=0.5,
    s=64.0
)
```

---

## Important Parameters

### `milestone0`

The epoch of the first learning rate decay or the epoch before which the progressive filtering schedule is expected to become stable.

Example:

```python
milestone0=11
```

### `gap_epoch`

Controls the progressive threshold schedule. The threshold gradually moves from the estimated lower bound to the upper bound before `milestone0`.

Example:

```python
gap_epoch=2
```

### `top_class`

The number of high non-target cosine similarities used to estimate the upper noise-related similarity bound.

Example:

```python
top_class=10
```

### `ema_t`

EMA update coefficient for smoothing the estimated statistics.

Example:

```python
ema_t=0.01
```

### `alpha`

A small slack term added when estimating the upper bound.

Example:

```python
alpha=0.02
```

---

## How It Works

At each training step, the module receives:

```python
cosine: Tensor of shape [B, C]
label:  Tensor of shape [B]
```

where:

- `B` is the batch size
- `C` is the number of identities/classes
- `cosine` is the cosine similarity between image features and class centers
- `label` is the ground-truth identity label

The module then:

1. Extracts the target-class cosine similarity of each sample.
2. Tracks noise-related similarity statistics with EMA.
3. Estimates a progressive threshold.
4. Filters samples whose target-class cosine similarity is below the current threshold.
5. Returns the filtered cosine similarities and labels.

Filtering is disabled at epoch 0 to avoid unstable early training.

---

## Dataset

The dataset used in our experiments is hosted on Baidu Netdisk.

```text
Baidu Netdisk: 填写你的百度网盘链接
Extraction code: 填写你的提取码
```

Please replace the placeholders above with the actual download link and extraction code.

---

## Dataset Usage Notice

The dataset is provided only for academic research purposes.

Users should follow all applicable laws, institutional requirements, and privacy regulations when using the dataset.

The following uses are not allowed:

- Commercial use without permission
- Redistribution without permission
- Any use that violates privacy or legal regulations
- Any use unrelated to academic research

---

## Recommended Dataset Structure

After downloading and extracting the dataset, the recommended structure is:

```text
dataset/
├── train/
│   ├── id_000001/
│   ├── id_000002/
│   ├── id_000003/
│   └── ...
├── val/
│   ├── id_000001/
│   ├── id_000002/
│   ├── id_000003/
│   └── ...
└── README.txt
```

If your dataset follows a different structure, please modify the data loading script accordingly.

---

## Integration into Existing Training Code

In a typical face recognition training pipeline, the model first extracts normalized face features. Then the classification layer computes cosine similarities between features and class centers.

The noise filter should be applied after cosine similarity computation and before the final margin-based loss computation.

A simplified training flow is:

```python
# 1. Extract face features
features = backbone(images)

# 2. Compute cosine similarities
cosine = classifier(features)

# 3. Apply noise filtering
cosine, labels = noise_filter(
    cosine,
    labels,
    current_epoch
)

# 4. Compute margin-based logits
logits, labels = margin_head(
    cosine,
    labels,
    current_epoch
)

# 5. Compute loss
loss = criterion(logits, labels)
```

If you use the provided `CosFace` or `ArcFace` classes with `denoiseEnable=True`, the filtering step is already included inside the head.

---

## Minimal Example with Training Loss

```python
import torch
import torch.nn as nn
from noise_filter_DDLN import build_head

B = 16
C = 1000

cosine = torch.randn(B, C)
label = torch.randint(0, C, (B,))

criterion = nn.CrossEntropyLoss()

head = build_head(
    head_type="cosface_denoise",
    m=0.35,
    s=64.0
)

current_epoch = 1

logits, label_new = head(
    cosine,
    label,
    current_epoch
)

loss = criterion(logits, label_new)

print(loss.item())
```

---

## Notes

- The module is designed for supervised face recognition training with noisy labels.
- The input `cosine` should represent cosine similarities between samples and class centers.
- The label tensor should contain integer class labels.
- Filtering is disabled at epoch 0 to keep early training stable.
- `milestone0` should usually be aligned with the first learning rate decay epoch.
- The method does not require prior knowledge of the noise rate.
- The method does not require an auxiliary network.

---

## Citation

If this repository is useful for your research, please cite our paper after it becomes available.

```bibtex
@article{your_paper_key,
  title   = {Your Paper Title},
  author  = {Your Name and Others},
  journal = {To appear},
  year    = {2026}
}
```

---

## License

This repository is currently released for academic research use.

Please contact the authors if you would like to use the code or dataset for other purposes.

---

## Contact

For questions about the code or dataset, please contact:

```text
Name: Your Name
Email: your_email@example.com
```
