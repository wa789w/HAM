# HAM
Official implementation of the AAAI'26 paper: "Beyond Single-Point Perturbation: A Hierarchical, Manifold-Aware Approach to Diffusion Attacks".

[[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/38013)

<p align="center">
  <img src="assets/overview.png" width="95%">
</p>

## Dataset Setup

### 1. ImageNet-Compatible Dataset
Download the **ImageNet-Compatible** dataset, unzip it, and place the images in the `images` directory.

The dataset should be organized as:
```
images/
├── 1.png
├── 2.png
├── ...
└── 1000.png

labels.txt  # Contains corresponding ImageNet class labels
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
We adopt **Stable Diffusion 2.0** as our latent diffusion model. Please download the model weights:
1. Download the `512-base-ema.ckpt` file
2. Place it in the `./ckpt/` folder

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
  --input_dir images \
  --label_file labels.txt \
  --ckpt ckpt/512-base-ema.ckpt \
  --output_dir output_adv \
  --apply_adv \
  --enable_grad \
  --start_step 12 \
  --adv_start 17 \
  --adv_end 28 \
  --adv_epsilon 0.035 \
  --target_model resnet50
```

The command above reads images from `images/`, labels from `labels.txt`, and the Stable Diffusion checkpoint from `ckpt/512-base-ema.ckpt`.

## Outputs

The output directory contains:

```text
output_adv/
├── adv/
│   ├── 1.png
│   ├── 2.png
│   └── ...
└── attack_results.csv
```

`attack_results.csv` records the original label, initial prediction, final prediction, confidence scores, attack success flag, and start step.

## Citation

If you find this project useful, please cite:

```bibtex
@inproceedings{wang2026beyond,
  title={Beyond Single-Point Perturbation: A Hierarchical, Manifold-Aware Approach to Diffusion Attacks},
  author={Wang, Zhijie and Wang, Lin and Wen, Zhenyu and Wang, Cong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={12},
  pages={10421--10429},
  year={2026}
}
```
