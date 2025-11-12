# HAM
Official implementation of the AAAI'26 paper: "Beyond Single-Point Perturbation: A Hierarchical, Manifold-Aware Approach to Diffusion Attacks".

## Dataset Setup

### 1. ImageNet-Compatible Dataset
Download the **ImageNet-Compatible** dataset, unzip it, and place it in the `imagenet/images` directory.

The dataset should be organized as:
```
imagenet/
├── images/
│   ├── 1.png
│   ├── 2.png
│   ├── ...
└── labels.txt  # Contains corresponding ImageNet class labels
```

### 2. Label Format
The `labels.txt` file should contain one label per line (1-indexed), corresponding to ImageNet class indices:
```
285
482
491
...
```

---

## Model Weights

### Stable Diffusion 2.0 Weights
we adopt **Stable Diffusion 2.0** as our latent diffusion model. Please download the model weights:
1. Download the `512-base-ema.ckpt` file
2. Place it in the `./checkpoints/` folder

### Alternative Model Support
The framework also supports other Stable Diffusion variants. Update the `--ckpt` parameter accordingly:
- `v2-1_512-ema-pruned.ckpt` for Stable Diffusion 2.1
- Custom fine-tuned models

---

## Usage

### Default Attack
Run the default adversarial attack on all images:

```bash
python main.py \
    --input_dir imagenet/images \
    --label_file imagenet/labels.txt \
    --output_dir output_adv \
    --apply_adv \
    --enable_grad \
    --target_model resnet50
```
