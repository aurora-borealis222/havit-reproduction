# HAViT Reproduction

Reproduction of **[arXiv:2603.18585](https://arxiv.org/abs/2603.18585)** (Banik et al., 2026)
with three architectural experiments on top of the authors' code.

Authors' original repository: [github.com/banik-s/HAViT](https://github.com/banik-s/HAViT)


## Repository structure

```
havit-reproduction/               <- this repo
├── experiments/
│   ├── run.py                    <- entry point for ALL experiments
│   └── models/
│       ├── havit_learnable_alpha.py   <- EXP-A
│       ├── havit_post_softmax.py      <- EXP-B
│       ├── havit_zero_init.py         <- EXP-C
│   -- HAViT_Colab.ipynb         <- Colab runner (imports from py files)
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Experiments

| # | Key | What changes                           | Research question |
|---|---|----------------------------------------|---|
| repro | `baseline_vit` | -                                      | Paper baseline: 75.74%? |
| repro | `havit_v1` | -                                      | Paper HAViT: 77.07%? |
| A | `learnable_alpha` | Fixed α=0.9 -> per-head `sigmoid(w_h)` | Can heads learn different α? |
| B | `post_softmax` | Blend logits -> blend probabilities    | Does the blending space matter? |
| C | `zero_init` | H₀=randn -> H₀=zeros                   | How much does init noise help? |

## How to run

### Colab (recommended)
Open `HAViT_reproduction.ipynb` in Colab with T4 GPU.
The notebook clones both repos, sets up paths and calls `experiments/run.py`.


## Logging
TensorBoard logs are written by the authors' `main.py` to `HAViT/tensorboard/`.
```bash
tensorboard --logdir HAViT/tensorboard
```

## Requirements
- Python ≥ 3.11
- PyTorch ≥ 2.0

## Code quality
```bash
black experiments/
isort experiments/
```
