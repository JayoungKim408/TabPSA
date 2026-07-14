import numpy as np
import torch
import torch.nn as nn
from models import TabTransformer_TabPSA, MLP
from dataloader import get_dataloader
from utils import EarlyStopping, basic_setting, set_seed
from sklearn.preprocessing import OneHotEncoder
from train import train_pretrained_model, train_mlp_model, test_model

def main():
    # 1. Basic Setting
    args, logger = basic_setting()
    set_seed(args.seed)
    logger.info(args)

    # 2. Datalodaer
    train_dataloader, val_dataloader, test_dataloader, dataset = get_dataloader(args, args.dataset.lower())
    categ_dims = dataset.categ_dims
    cont_index = dataset.cont_index

    if args.task in ['binary', 'multi']:
        onehot_encoder = OneHotEncoder(sparse_output=False)
        onehot_encoder.fit(dataset.train_y)
        num_target = len(onehot_encoder.categories_[0])
    else:
        onehot_encoder = None
        num_target = 1

    # 3. Pretrain
    # 3.1 pretrain model
    pretrained_model = TabTransformer_TabPSA( 
        categories = categ_dims, 
        num_continuous = len(cont_index),
        dim = args.dim, 
        dim_out = sum(categ_dims)+len(cont_index), 
        depth = args.depth,
        heads = args.n_heads,
        attn_dropout = args.attn_dropout,
        ff_dropout = args.ff_dropout,
        mlp_hidden_mults = (4, 2),
        K = args.K,
        polynomial = args.polynomial
    ).to(args.device)
    logger.info(pretrained_model)
    logger.info(f"# params of pretrained model: {sum(p.numel() for p in pretrained_model.parameters() if p.requires_grad)}\n")
    
    ## 3.2 pretrain
    train_pretrained_model(args, pretrained_model, train_dataloader, val_dataloader, test_dataloader, logger)
    torch.save(pretrained_model.state_dict(), args.pretrain_checkpoint_path)
    
    # 4. Downstream Task (Finetune)
    ## 4.1 mlp model
    set_seed(args.seed)
    mlp_input = (args.dim * len(categ_dims)) + len(cont_index)
    mlp_model = MLP(dims=(mlp_input, args.hidden_mlp, args.hidden_mlp, num_target)).to(args.device)
    logger.info(mlp_model)
    logger.info(f"# params of mlp model: {sum(p.numel() for p in mlp_model.parameters() if p.requires_grad)}\n")
    
    ## 4.2 load pretrained model
    pretrained_model.load_state_dict(torch.load(args.pretrain_checkpoint_path))
    logger.info(f"Load best pretrained model from {args.pretrain_checkpoint_path}")

    ## 4.3 finetune
    logger.info(f"Start finetuning mlp model")
    train_mlp_model(args, pretrained_model, mlp_model, train_dataloader, val_dataloader, test_dataloader, logger)

    # 5. Evaluation
    ## 5.1 load best pretrained model & mlp model
    pretrained_model.load_state_dict(torch.load(args.finetune_checkpoint_path))
    logger.info(f"Load best pretrained model from {args.finetune_checkpoint_path}")
    mlp_model.load_state_dict(torch.load(args.mlp_checkpoint_path))
    logger.info(f"Load best mlp model from {args.mlp_checkpoint_path}")

    ## 5.2 test model
    test_model(args, pretrained_model, mlp_model, test_dataloader, logger, onehot_encoder=onehot_encoder)

if __name__ == "__main__":
    main()
