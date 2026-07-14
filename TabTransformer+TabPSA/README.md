This is the official implementation for the paper [Polynomial-based Self-Attention for Table Representation learning]

--------------------

## How to run the code

### Dependencies

Run the following to install a subset of necessary python packages for our code
```sh
conda env create -f tabtransformer_env.yaml
```

### Usage

Train and evaluate our models through `main.py`.

```sh
train.py:
  --dataset: Dataset to train.
  --task: Task to perform (binary|multi|regression)
```

You can train TabTransformer+TabPSA by excuting the following command:
```
python main.py --dataset phishing --task binary
```
--------------------