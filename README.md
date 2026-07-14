# LAD 3D Model Implementations

A lightweight research-code repository that currently contains model definitions for 3D volumetric inputs.

- `LAD_UNETR.py`
- `LAD_SwinUNETR.py`
- `LAD_Primus.py`


## Repository Structure

```text
LAD_UNETR.py        # UNETR-style backbone; shared modules and base model
LAD_SwinUNETR.py # SwinUNETR-style backbone; depends on LAD_UNETR.py
LAD_Primus.py    # Primus / EVA-02-style backbone; depends on LAD_UNETR.py
README.md
```

`LAD_SwinUNETR.py` and `LAD_Primus.py` import shared components from `LAD_UNETR.py`. Direct imports work when running from the source directory; this repo is not packaged.

## Requirements

- Python >= 3.8
- PyTorch

Recommended setup:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# Or use the PyTorch build that matches your local CUDA setup
```

## Quick Start

These examples assume you have cloned the repository and are running from the directory containing these three `.py` files.

### UNETR-style

```python
import torch
from LAD_UNETR import LAD_UNETR

model = LAD_UNETR(
    in_channels=1,
    out_channels=2,
    img_size=(96, 96, 96),
    patch_size=16,
    task="segmentation",
)
x = torch.randn(2, 1, 96, 96, 96)
logits = model(x)
print(logits.shape)
```

### SwinUNETR-style

```python
import torch
from LAD_SwinUNETR import LAD_SwinUNETR

model = LAD_SwinUNETR(
    in_channels=1,
    out_channels=3,
    patch_size=2,
    task="segmentation",
)
x = torch.randn(1, 1, 96, 96, 96)
logits = model(x)
print(logits.shape)
```

### Primus-style

```python
import torch
from LAD_Primus import LAD_Primus

model = LAD_Primus(
    in_channels=1,
    out_channels=1,
    img_size=(96, 96, 96),
    patch_size=(8, 8, 8),
    task="segmentation",
)
x = torch.randn(1, 1, 96, 96, 96)
logits = model(x)
print(logits.shape)
```
