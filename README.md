<h1 align="center">HAM</h1>

<h2 align="center">
  <a href="https://ojs.aaai.org/index.php/AAAI/article/view/38013">Beyond Single-Point Perturbation: A Hierarchical, Manifold-Aware Approach to Diffusion Attacks</a>
</h2>

<p align="center">
  <b>Official implementation of the AAAI'26 paper</b>
</p>

<p align="center">
  <img src="assets/overview.png" width="95%">
</p>

## Installation

Create the environment with Conda:

```bash
conda env create -f environment.yml
conda activate HAM
```

The environment file installs the core PyTorch, Stable Diffusion, and evaluation dependencies required by HAM.

## Dataset Setup

### 1. ImageNet-Compatible Dataset
HAM uses 1,000 ImageNet-compatible images stored in the `images` directory.

The dataset should be organized as:
```
images/
├── 1.png
├── 2.png
├── ...
└── 1000.png

labels.txt  # Contains corresponding ImageNet class labels
```

The default command reads images directly from `images/` and labels from `labels.txt`.

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
HAM uses **Stable Diffusion 2.0 base**, following prior diffusion-based attack studies. Since the original Stability AI repository is no longer publicly accessible, we provide a compatible Hugging Face backup link:

- Model page: [Manojb/stable-diffusion-2-base](https://huggingface.co/Manojb/stable-diffusion-2-base)
- Direct checkpoint link: [512-base-ema.ckpt](https://huggingface.co/Manojb/stable-diffusion-2-base/resolve/main/512-base-ema.ckpt)

Place the checkpoint under:

```text
ckpt/
└── 512-base-ema.ckpt
```

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
