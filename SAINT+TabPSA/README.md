# Polynomial-based Self-Attention for Table Representation Learning


This is the official implementation for the paper [Polynomial-based Self-Attention for Table Representation learning]

--------------------

## How to run the code

### Dependencies

Run the following to install a subset of necessary python packages for our code
```sh
conda env create -f requirements.yaml
```

### Usage

Train and evaluate our models through `saint_tabpsa.py`.

```sh
saint_tabpsa.py:
  --dataset_name: Dataset to train.
```

By run the following script, you can reproduce the result reperted in the original paper.
```
python saint_tabpsa.py --pretrain  --task binary --dataset_name phishing --polynomial chebyshev --lr 0.005 --embedding_size 16 --transformer_depth 6 --attention_heads 4 --pt_aug_lam 0.1 --mixup_lam 0.3 --alpha -0.99 --beta 0.0 --nce_temp 0.7 --final_mlp_style sep --K 5

```
--------------------
