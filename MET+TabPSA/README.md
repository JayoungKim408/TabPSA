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

Train and evaluate our models through `met_tabpsa.py`.

```sh
met_tabpsa.py:
  --dataset_name: Dataset to train.
```

By run the following script, you can reproduce the result reperted in the original paper.
```
python met_tabpsa.py --dataset_name phishing --embed_dim 64 --num_heads 3 --encoder_depth 3 --decoder_depth 3 --lr 0.005 --radius 2 --K 3 --clf_lr 0.001 --polynomial jacobi --alpha 1.0 --beta 0.3
```
--------------------
