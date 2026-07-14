import os
import numpy as np
import torch
import torch.nn as nn
from utils import EarlyStopping
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score, mean_squared_error
from scipy.special import softmax

def train_pretrained_model(args, model, trainloader, valloader, testloader, logger):

    categories = model.categories
    cat_index = np.concatenate([[0], np.cumsum(categories)])

    device = args.device
    categ_loss_fn = nn.CrossEntropyLoss()
    cont_loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(args.pretrain_checkpoint_path, logger=logger, patience=args.early_stopping_rounds, verbose=True)

    for ep in range(args.epoch):
        model.train()
        train_loss = 0.0
        for iter, batch in enumerate(trainloader):

            batch = tuple(t.float().to(device) for t in batch)
            x_categ, x_cont, _, x_categ_masked, x_cont_masked, categ_mask, cont_mask = batch 
            
            output = model(x_categ_masked, x_cont_masked)

            categ_loss = 0.0
            for k in range(x_categ.shape[-1]):
                exist_mask = torch.isin(categ_mask, torch.tensor(k, device=args.device)).sum(axis=1).bool()
                if exist_mask.sum() > 0:
                    k_y = x_categ[exist_mask, k]  # +1 for masked embedding in data preprocessing
                    k_logit_index = np.arange(cat_index[k], cat_index[k+1])
                    k_logit_index = np.repeat([k_logit_index], k_y.shape[0], axis=0)
                    k_logits = torch.gather(output[exist_mask], 1, torch.tensor(k_logit_index, device=output.device))
                    categ_loss += categ_loss_fn(k_logits, k_y.long())

            cont_loss = 0.0
            for k in range(x_cont.shape[-1]):
                exist_mask = torch.isin(cont_mask, torch.tensor(k, device=args.device)).sum(axis=1).bool()
                if exist_mask.sum() > 0:
                    k_y = x_cont[exist_mask, k]
                    k_pred = output[exist_mask, -(k+1)]
                    cont_loss += cont_loss_fn(k_y, k_pred)

            loss = args.categ_loss_ratio * categ_loss + cont_loss 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # logging
        post_fix = {
            "epoch": ep,
            "train_loss": '{:.4f}'.format(train_loss / len(trainloader)),
        }
        logger.info(str(post_fix))

        # VALIDATION
        if args.early_stopping_rounds > 0:
            if validate_pretrained_model(args, model, valloader, device, early_stopping, logger, [categ_loss_fn, cont_loss_fn], ep):
                break

def validate_pretrained_model(args, model, valloader, device, early_stopping, logger, loss_fn, ep):

    categ_loss_fn, cont_loss_fn = loss_fn
    categories = model.categories
    cat_index = np.concatenate([[0], np.cumsum(categories)])

    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for iter, batch in enumerate(valloader):
            batch = tuple(t.to(device).float() for t in batch)
            x_categ, x_cont, _, x_categ_masked, x_cont_masked, categ_mask, cont_mask = batch
            output = model(x_categ_masked, x_cont_masked)
            
            categ_loss = 0.0
            for k in range(x_categ.shape[-1]):
                exist_mask = torch.isin(categ_mask, torch.tensor(k, device=args.device)).sum(axis=1).bool()
                if exist_mask.sum() > 0:
                    k_y = x_categ[exist_mask, k]  # +1 for masked embedding in data preprocessing
                    k_logit_index = np.arange(cat_index[k], cat_index[k+1])
                    k_logit_index = np.repeat([k_logit_index], k_y.shape[0], axis=0)
                    k_logits = torch.gather(output[exist_mask], 1, torch.tensor(k_logit_index, device=output.device))
                    categ_loss += categ_loss_fn(k_logits, k_y.long())

            cont_loss = 0.0
            for k in range(x_cont.shape[-1]):
                exist_mask = torch.isin(cont_mask, torch.tensor(k, device=args.device)).sum(axis=1).bool()
                if exist_mask.sum() > 0:
                    k_y = x_cont[exist_mask, k]
                    k_pred = output[exist_mask, -(k+1)]
                    cont_loss += cont_loss_fn(k_y, k_pred)

            loss = args.categ_loss_ratio * categ_loss + cont_loss
            val_loss += loss.item()

        early_stopping(-np.array([val_loss/len(valloader)]), model)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            return True

    return False

def train_mlp_model(args, pretrained_model, mlp_model, trainloader, valloader, testloader, logger):

    device = args.device
    loss_fn = nn.MSELoss() if args.task=='regression' else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(pretrained_model.parameters())+list(mlp_model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)

    early_stopping = EarlyStopping([args.finetune_checkpoint_path, args.mlp_checkpoint_path], logger=logger, patience=args.early_stopping_rounds, verbose=True)

    for ep in range(args.epoch):

        pretrained_model.train()
        mlp_model.train()

        train_loss = 0.0
        for iter, batch in enumerate(trainloader):

            batch = tuple(t.to(device).float() for t in batch)
            x_categ, x_cont, y, _, _, _, _ = batch
            y = y if args.task == 'regression' else y.squeeze().long()

            _, attentions, representation = pretrained_model(x_categ, x_cont, True)

            output = mlp_model(representation)

            loss = loss_fn(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # logging
        post_fix = {
            "epoch": ep,
            "train_loss": '{:.4f}'.format(train_loss / len(trainloader)),
        }
        logger.info(str(post_fix))

        # VALIDATION
        if args.early_stopping_rounds > 0:
            if validate_mlp_model(args, pretrained_model, mlp_model, valloader, device, early_stopping, logger, loss_fn, ep):
                break

def validate_mlp_model(args, pretrained_model, mlp_model, valloader, device, early_stopping, logger, loss_fn, ep):

    pretrained_model.eval()
    mlp_model.eval()

    with torch.no_grad():
        val_loss = 0.0
        for iter, batch in enumerate(valloader):
            batch = tuple(t.to(device).float() for t in batch)
            x_categ, x_cont, y, _, _, _, _ = batch
            y = y if args.task == 'regression' else y.squeeze().long()
            
            _, _, representation = pretrained_model(x_categ, x_cont, True)
            output = mlp_model(representation)

            loss = loss_fn(output, y)
            val_loss += loss.item()

        early_stopping(-np.array([val_loss/len(valloader)]), [pretrained_model, mlp_model], True)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            return True
        
    return False

def test_model(args, pretrained_model, mlp_model, testloader, logger, onehot_encoder=None):
    device = args.device
    y_pred_collector = []
    y_true_collector = []
    
    before_dict = {}
    after_dict = {}
    feat_dict = {}
    tx_dict = {}
    for d in range(args.depth):
        before_dict[d] = []
        after_dict[d] = []
        feat_dict[d] = []
        tx_dict[d] = [[],[],[],[],[],[]]
        
    with torch.no_grad():
        for iter, batch in enumerate(testloader):
            pretrained_model.eval()
            mlp_model.eval()

            batch = tuple(t.to(device) for t in batch)
            x_categ, x_cont, y, _, _, _, _ = batch

            _, attentions, representation = pretrained_model(x_categ, x_cont, True)
            output = mlp_model(representation)

            y_pred_collector.append(output)
            y_true_collector.append(y)

    y_pred = torch.concat(y_pred_collector).cpu().detach().numpy()
    y_true = torch.concat(y_true_collector).cpu().detach().numpy()

    if args.task in ['binary', 'multi']:
        if y_true.max() == 1: # binary
            acc = accuracy_score(y_true, y_pred.argmax(axis=1))
            auc = roc_auc_score(y_true, softmax(y_pred[:, 1]))
        else: # multi-class
            y_encoded = onehot_encoder.transform(y_true.reshape(-1, 1))
            acc = accuracy_score(y_true, y_pred.argmax(axis=1))
            auc = roc_auc_score(y_encoded, y_pred, multi_class='ovr')

        logger.info(f"Test Result = Accuracy: [{acc}], ROC_AUC: [{auc}]")
    else:
        r2 = r2_score(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False) # setting squared False for computing RMSE

        logger.info(f"Test Result = R2: [{r2}], RMSE: [{rmse}]")
