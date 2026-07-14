# Polynomial-based Self-Attention for Table Representation Learning (TabPSA)

This is the official implementation for the paper **Polynomial-based Self-Attention for Table Representation Learning**.

We integrate the polynomial-based self-attention (TabPSA) module into three tabular transformer backbones. Each backbone lives in its own directory with its own environment and entry point:

| Backbone | Directory | Entry point | Environment file |
|---|---|---|---|
| MET | [`MET+TabPSA/`](MET+TabPSA/) | `met_tabpsa.py` | `requirements.yaml` |
| SAINT | [`SAINT+TabPSA/`](SAINT+TabPSA/) | `saint_tabpsa.py` | `saint_environment.yml` |
| TabTransformer | [`TabTransformer+TabPSA/`](TabTransformer+TabPSA/) | `main.py` | `tabtransformer_env.yaml` |

Datasets shared across all backbones are in [`data/`](data/).

---

## Repository structure

```
.
├── data/                    # Shared datasets (CSV/TXT)
├── MET+TabPSA/              # MET backbone + TabPSA
├── SAINT+TabPSA/           # SAINT backbone + TabPSA
└── TabTransformer+TabPSA/  # TabTransformer backbone + TabPSA
```

---

## Setup

Each backbone has its own conda environment. Create the one you need from inside its directory:

```sh
# MET
cd MET+TabPSA && conda env create -f requirements.yaml

# SAINT
cd SAINT+TabPSA && conda env create -f saint_environment.yml

# TabTransformer
cd TabTransformer+TabPSA && conda env create -f tabtransformer_env.yaml
```

---

## Usage

### MET + TabPSA

Train and evaluate through `met_tabpsa.py`. To reproduce the result reported in the paper:

```sh
python met_tabpsa.py --dataset_name phishing --embed_dim 64 --num_heads 3 \
  --encoder_depth 3 --decoder_depth 3 --lr 0.005 --radius 2 --K 3 \
  --clf_lr 0.001 --polynomial jacobi --alpha 1.0 --beta 0.3
```

### SAINT + TabPSA

Train and evaluate through `saint_tabpsa.py`. To reproduce the result reported in the paper:

```sh
python saint_tabpsa.py --pretrain --task binary --dataset_name phishing \
  --polynomial chebyshev --lr 0.005 --embedding_size 16 --transformer_depth 6 \
  --attention_heads 4 --pt_aug_lam 0.1 --mixup_lam 0.3 --alpha -0.99 --beta 0.0 \
  --nce_temp 0.7 --final_mlp_style sep --K 5
```

### TabTransformer + TabPSA

Train and evaluate through `main.py`:

```sh
python main.py --dataset phishing --task binary
```

Common arguments: `--dataset` selects the dataset, and `--task` is one of `binary`, `multi`, or `regression`.
