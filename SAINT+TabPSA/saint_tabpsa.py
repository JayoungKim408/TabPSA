import torch
from torch import nn
from models import SAINT
import time
from data import load_data, DataSetCatCon
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import count_parameters, classification_scores, mean_sq_error, r2
from augmentations import embed_data_mask
from augmentations import add_noise
from datasets import TabularDataset

import os
import numpy as np
import pandas as pd
parser = argparse.ArgumentParser()

parser.add_argument('--dataset_name', default='phishing', type=str)
parser.add_argument('--runname', default='saint_tabpsa', type=str)
parser.add_argument('--task', default = 'binary', type=str,choices = ['binary','multiclass','regression'])
parser.add_argument('--vision_dset', action = 'store_true')
parser.add_argument('--cont_embeddings', default='MLP', type=str,choices = ['MLP','Noemb','pos_singleMLP'])
parser.add_argument('--embedding_size', default=32, type=int)
parser.add_argument('--transformer_depth', default=1, type=int)
parser.add_argument('--attention_heads', default=4, type=int)
parser.add_argument('--attention_dropout', default=0.1, type=float)
parser.add_argument('--ff_dropout', default=0.2, type=float)
parser.add_argument('--attentiontype', default='colrow', type=str,choices = ['col','colrow','row','justmlp','attn','attnmlp'])

parser.add_argument('--optimizer', default='Adam', type=str,choices = ['AdamW','Adam','SGD'])
parser.add_argument('--scheduler', default='cosine', type=str,choices = ['cosine','linear'])

parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batchsize', default=256, type=int)
parser.add_argument('--savemodelroot', default='./bestmodels', type=str)
parser.add_argument('--set_seed', default= 1 , type=int)
parser.add_argument('--dset_seed', default= 5 , type=int)
parser.add_argument('--active_log', action = 'store_true')

parser.add_argument('--pretrain', action = 'store_true')
parser.add_argument('--pretrain_epochs', default=100, type=int)
parser.add_argument('--pt_tasks', default=['contrastive','denoising'], type=str,nargs='*',choices = ['contrastive','contrastive_sim','denoising'])
parser.add_argument('--pt_aug', default=['mixup','cutmix'], type=str, nargs='*',choices = ['mixup','cutmix'])
parser.add_argument('--pt_aug_lam', default=0.1, type=float)
parser.add_argument('--mixup_lam', default=0.3, type=float)

parser.add_argument('--train_mask_prob', default=0, type=float)
parser.add_argument('--mask_prob', default=0, type=float)

parser.add_argument('--ssl_avail_y', default= 0, type=int)
parser.add_argument('--pt_projhead_style', default='diff', type=str,choices = ['diff','same','nohead'])
parser.add_argument('--nce_temp', default=0.7, type=float)

parser.add_argument('--lam0', default=0.5, type=float)
parser.add_argument('--lam1', default=10, type=float)
parser.add_argument('--lam2', default=1, type=float)
parser.add_argument('--lam3', default=10, type=float)
parser.add_argument('--final_mlp_style', default='sep', type=str,choices = ['common','sep'])
parser.add_argument('--polynomial', default='power', type=str,choices = ['original', 'chebyshev','power', 'legendre', 'jacobi'])

parser.add_argument('--alpha', default=1.0, type=float)
parser.add_argument('--beta', default=0.3, type=float)
parser.add_argument('--K', default=3, type=int)

opt = parser.parse_args()
from datetime import datetime

run_name = f"{opt.polynomial}_{opt.lr}_{opt.embedding_size}_{opt.transformer_depth}_{opt.attention_heads}_{opt.pt_aug_lam}_{opt.mixup_lam}_{opt.alpha}_{opt.beta}_{opt.nce_temp}_{opt.final_mlp_style}_{opt.K}"
save_path = f"{opt.savemodelroot}/{opt.task}/{opt.dataset_name}/{run_name}.pth"
curr_time = datetime.now().strftime("%b%d_%H%M%S")
modelsave_path = os.path.join(os.getcwd(), opt.savemodelroot, opt.task, str(opt.dataset_name))
if opt.task == 'regression':
    opt.dtask = 'reg'
else:
    opt.dtask = 'clf'


device = 'cuda' if torch.cuda.is_available() else "cpu" # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}.")

os.makedirs(modelsave_path, exist_ok=True)

if opt.active_log:
    import wandb
    wandb.init(project = "saint_tabpsa", group =opt.dataset_name ,name = run_name)

print('Downloading and processing the dataset, it might take some time.')

# best_test_score_5_times = []
# for _ in range(5):
cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(opt.dataset_name, opt.dset_seed)
continuous_mean_std = None

##### Setting some hyperparams based on inputs and dataset
_,nfeat = X_train['data'].shape
if nfeat > 100:
    opt.embedding_size = min(8,opt.embedding_size)
    opt.batchsize = min(64, opt.batchsize)
if opt.attentiontype != 'col':
    opt.transformer_depth = 1
    opt.attention_heads = min(4,opt.attention_heads)
    opt.attention_dropout = 0.8
    opt.embedding_size = min(32,opt.embedding_size)
    opt.ff_dropout = 0.8

print(nfeat,opt.batchsize)
print(opt)

if opt.active_log:
    wandb.config.update(opt)
train_ds = DataSetCatCon(X_train, y_train, cat_idxs,opt.dtask, continuous_mean_std)
trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True,num_workers=4)

valid_ds = DataSetCatCon(X_valid, y_valid, cat_idxs,opt.dtask, continuous_mean_std)
validloader = DataLoader(valid_ds, batch_size=opt.batchsize, shuffle=False,num_workers=4)

test_ds = DataSetCatCon(X_test, y_test, cat_idxs,opt.dtask, continuous_mean_std)
# testloader = DataLoader(test_ds, batch_size=X_test['data'].shape[0], shuffle=False,num_workers=4)
testloader = DataLoader(test_ds, batch_size=1000, shuffle=False,num_workers=4)
if opt.task == 'regression':
    y_dim = 1
else:
    y_dim = len(np.unique(y_train['data'][:,0]))

cat_dims = np.append(np.array([1]),np.array(cat_dims)).astype(int) #Appending 1 for CLS token, this is later used to generate embeddings.



model = SAINT(
    categories = tuple(cat_dims), 
    num_continuous = len(con_idxs),                
    dim = opt.embedding_size,                           
    dim_out = 1,  
    K=opt.K,
    depth = opt.transformer_depth,                       
    heads = opt.attention_heads,                         
    attn_dropout = opt.attention_dropout,             
    ff_dropout = opt.ff_dropout,                  
    mlp_hidden_mults = (4, 2),       
    cont_embeddings = opt.cont_embeddings,
    attentiontype = opt.attentiontype,
    final_mlp_style = opt.final_mlp_style,
    y_dim = y_dim,
    alpha=opt.alpha,
    beta=opt.beta, 
    polynomial=opt.polynomial
)
vision_dset = opt.vision_dset


if y_dim == 2 and opt.task == 'binary':
    criterion = nn.CrossEntropyLoss().to(device)
elif y_dim > 2 and  opt.task == 'multiclass':
    criterion = nn.CrossEntropyLoss().to(device)
elif opt.task == 'regression':
    criterion = nn.MSELoss().to(device)
else:
    raise'case not written yet'

model.to(device)
from pretraining import SAINT_pretrain
model = SAINT_pretrain(model, cat_idxs, X_train, y_train, opt, device, continuous_mean_std)
    
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model)
print(f"The number of model parameters: {num_params}")

## Choosing the optimizer

if opt.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=opt.lr,
                        momentum=0.9, weight_decay=5e-4)
    from utils import get_scheduler
    scheduler = get_scheduler(opt, optimizer)
elif opt.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(),lr=opt.lr)
elif opt.optimizer == 'AdamW':
    optimizer = optim.AdamW(model.parameters(),lr=opt.lr)

best_valid_auroc = 0
best_valid_accuracy = 0
best_test_auroc = 0
best_test_accuracy = 0
best_valid_rmse = 100000
best_valid_r2 = 0
best_test_r2 = 0
print('Training begins now.')
training_times = []
for epoch in range(opt.epochs):
    training_start = time.time()
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        optimizer.zero_grad()
        # x_categ is the the categorical data, x_cont has continuous data, y_gts has ground truth ys. cat_mask is an array of ones same shape as x_categ and an additional column(corresponding to CLS token) set to 0s. con_mask is an array of ones same shape as x_cont. 
        x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)
        # We are converting the data to embeddings in the next step
        _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)           
        reps = model.transformer(x_categ_enc, x_cont_enc)
        # select only the representations corresponding to CLS token and apply mlp on it in the next step to get the predictions.
        y_reps = reps[:,0,:]
        
        y_outs = model.mlpfory(y_reps)
        if opt.task == 'regression':
            loss = criterion(y_outs,y_gts) 
        else:
            y_gts = y_gts.to(int)
            loss = criterion(y_outs,y_gts.squeeze()) 
        loss.backward()
        optimizer.step()
        if opt.optimizer == 'SGD':
            scheduler.step()
        running_loss += loss.item()
    training_end = time.time()
    training_times.append(training_end-training_start)
    if opt.active_log:
        wandb.log({'epoch': epoch ,'train_epoch_loss': running_loss, 
        'loss': loss.item()
        })

    if epoch%5==0:
    # if epoch >= 5:
        model.eval()
        times =[]
        for i in range(6):
            with torch.no_grad():
                if opt.task in ['binary','multiclass']:
                    accuracy, auroc, _ = classification_scores(model, validloader, device, opt.task,vision_dset)
                    test_accuracy, test_auroc, time_ = classification_scores(model, testloader, device, opt.task,vision_dset)
                    times.append(time_)
                    print('[EPOCH %d] VALID ACCURACY: %.3f, VALID AUROC: %.3f' %
                        (epoch + 1, accuracy,auroc ))
                    print('[EPOCH %d] TEST ACCURACY: %.3f, TEST AUROC: %.3f' %
                        (epoch + 1, test_accuracy,test_auroc ))
                    if opt.active_log:
                        wandb.log({'valid_accuracy': accuracy ,'valid_auroc': auroc })     
                        wandb.log({'test_accuracy': test_accuracy ,'test_auroc': test_auroc })  
                    if opt.task =='multiclass':
                        if accuracy > best_valid_accuracy:
                            best_valid_accuracy = accuracy
                            best_test_auroc = test_auroc
                            best_test_accuracy = test_accuracy
                            torch.save(model.state_dict(),'%s/%s.pth' % (modelsave_path, curr_time))
                    else:
                        if accuracy > best_valid_accuracy:
                            best_valid_accuracy = accuracy
                            best_test_auroc = test_auroc
                            best_test_accuracy = test_accuracy               
                            torch.save(model.state_dict(),'%s/%s.pth' % (modelsave_path, curr_time))

                else:
                    valid_r2, _ = r2(model, validloader, device,vision_dset)    
                    test_r2, time_ = r2(model, testloader, device,vision_dset)  
                    times.append(time_)

                    print('[EPOCH %d] VALID r2: %.3f' %
                        (epoch + 1, valid_r2 ))
                    print('[EPOCH %d] TEST r2: %.3f' %
                        (epoch + 1, test_r2 ))
                    if opt.active_log:
                        wandb.log({'valid_r2': valid_r2 ,'test_r2': test_r2 })     
                    if valid_r2 > best_valid_r2:
                        best_valid_r2 = valid_r2
                        best_test_r2 = test_r2
                        torch.save(model.state_dict(),'%s/%s.pth' % (modelsave_path, curr_time))
        model.train()

total_parameters = count_parameters(model)
print('TOTAL NUMBER OF PARAMS: %d' %(total_parameters))
if opt.task =='binary' or opt.task == "multiclass":
    print(f"AUROC on best model: {best_test_auroc}")
else:
    print(f"RMSE on best model: {best_test_r2}")

if opt.active_log:
    if opt.task == 'regression':
        wandb.log({'total_parameters': total_parameters, 'test_r2_bestep_5_times':best_test_r2, 'cat_dims':len(cat_idxs) , 'con_dims':len(con_idxs) })        
    else:
        wandb.log({'total_parameters': total_parameters, 'test_auroc_bestep_5_times':best_test_auroc , 'test_accuracy_bestep':best_test_accuracy,'cat_dims':len(cat_idxs) , 'con_dims':len(con_idxs) })
