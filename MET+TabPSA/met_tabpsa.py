# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from silence_tensorflow import silence_tensorflow
# silence_tensorflow()
import os
import time
import contextlib
from typing import Sequence, Any, ContextManager
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
from absl import app
from absl import flags
from absl import logging
from einops import rearrange
from torch.utils.data import DataLoader
import wandb
from torch import nn, einsum
import torch.nn.functional as F
from polynomials import * 
from pytorchtools import EarlyStopping

import numpy as np
import pandas as pd
import torch
from datasets import TabularDataset

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_name', 'phishing', 'dataset name')
flags.DEFINE_integer('embed_dim', 64, 'Embedding Dimension')
flags.DEFINE_integer('ff_dim', 32, 'FF Dimension')
flags.DEFINE_integer('num_heads', 8, 'Num heads')
flags.DEFINE_integer('encoder_depth', 3, 'Num Encoder Layers')
flags.DEFINE_integer('decoder_depth', 6, 'Num Decoder Layers')
flags.DEFINE_integer('mask_pct', 80, 'Mask Pct')
flags.DEFINE_float('lr',1e-04,'Learning rate')
flags.DEFINE_string('model_path','', 'Path for saved model')
flags.DEFINE_string('save_path','./tabpsa/', 'Path for saved model')
flags.DEFINE_string('run_name',"original", 'wanbd run name')
flags.DEFINE_integer('radius',2, 'Radius')
flags.DEFINE_float('lr_adv',1e-01,'Learning rate')
flags.DEFINE_integer('adv_steps',1, 'Adversatial steps')
flags.DEFINE_float('clf_lr',1e-03,'Learning rate')
flags.DEFINE_string('model_path_linear','', 'Path for saved linear model')
flags.DEFINE_string('polynomial','chebyshev', 'Polynomial type')
flags.DEFINE_boolean('active_log',False, 'wandb')
flags.DEFINE_bool('enc_cheb', True, 'encoder cheb')
flags.DEFINE_bool('dec_cheb', False, 'decoder cheb')
flags.DEFINE_bool('visualize', False, 'export attention')
flags.DEFINE_integer('K', 5, 'K')
flags.DEFINE_float('alpha', 1.0, 'jacobi polynimoal parameter alpha')
flags.DEFINE_float('beta', 0.3, 'jacobi polynimoal parameter beta')

rng = np.random.default_rng()
np.random.seed(42)
torch.manual_seed(42)

def mask_and_ind(arr, mask_pct=0.15):
    """Mask a given array unformly and randomly and return non-masked part of array, non-masked indices, masked indices"""
    r, c = arr.shape
    new_c = ((100-mask_pct)*c)//100
    rem_c = c - new_c
    shuff_idx = np.array([rng.permutation(c) for _ in range(r)])
    rem_idx = shuff_idx[:, :rem_c]
    new_idx = shuff_idx[:, rem_c:]
    new_idx.sort(axis=1)
    rem_idx.sort(axis=1)
    new_arr = np.ones((r, new_c))
    for i in range(r):
        new_arr[i] = arr[i][new_idx[i]]
    return new_arr, new_idx, rem_idx


def make_data(data, mask_pct):
    indexes = [i for i in range(len(data))]
    data = data.loc[indexes].iloc[:,:-1]
    arr = data.to_numpy()
    new_arr, new_idx, rem_idx = mask_and_ind(arr, mask_pct)
    small_maxlen = ((100-mask_pct)*data.shape[-1])//100

    new_arrs = []
    new_idxs = []
    rem_idxs = []
    ones = []
    ys = []

    for i in range(len(new_arr)):
        new_arrs.append(new_arr[i])
        new_idxs.append(new_idx[i])
        rem_idxs.append(rem_idx[i])
        ones.append(np.ones(small_maxlen))
        ys.append(arr[i][list(new_idx[i])+list(rem_idx[i])]) # 원래 데이터에서 순서 바뀜

    new_arrs=torch.tensor(new_arrs)
    new_idxs=torch.tensor(new_idxs, dtype=torch.long)
    rem_idxs=torch.tensor(rem_idxs, dtype=torch.long)
    ones=torch.tensor(ones, requires_grad=False)
    ys=torch.tensor(ys)

    return torch.concat([new_arrs, new_idxs, rem_idxs, ones, ys], axis=1)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# attention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)
    
class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, ff_dim),
            # GEGLU(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
        )

    def forward(self, x, **kwargs):
        return self.net(x)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        polynomial,
        heads = 8,
        dim_head = 16,
        dropout = 0.1, 
        cheb=False,
        K=5,
        alpha=1.0,
        beta=0.3
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.polynomial = polynomial

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.cheb = cheb
        
        if polynomial != 'original':
            if cheb:
                self.chebnet = PolyConvFrame(polynomial, depth = K, alpha = alpha, beta=beta, fixed = False)
            else: self.chebnet = None
        else:
            self.chebnet = None

    def forward(self, x, attention_mask=None):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        attention_scores = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale # attantion

        attention_scores = attention_scores.softmax(dim = -1)

        if self.chebnet:
            out = self.chebnet(x=v, adj=attention_scores)
        else:
            attn = self.dropout(attention_scores)
            out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)




class TokenAndPositionEmbedding(nn.Module):
  def __init__(self, maxlen, embed_dim):
    super().__init__()

    self.embed_dim = embed_dim
    self.pos_emb = nn.Embedding(maxlen, self.embed_dim) 

  def forward(self, x, positions_unmask, positions_mask):

    positions_unmask = self.pos_emb(positions_unmask)
    if positions_mask.shape[1]>=2:
      positions_mask = self.pos_emb(positions_mask)
    else:
      positions_mask = []
    x = x.reshape(x.shape[0], -1, 1)
    x = x.to(torch.float32)
    x = torch.cat([x, positions_unmask], dim=2)
    return x, positions_mask

# transformer
class TransformerBlock(nn.Module):
    def __init__(self, dim, polynomial, depth, heads, ff_dim, dim_head, ff_dropout, K, cheb=False, alpha=1.0, beta=0.3):
        super().__init__()
        self.layers = nn.ModuleList([])
        # self.layers = []
        self.dim = dim
        for _ in range(depth):
            self.layers.append( nn.ModuleList([
                PreNorm(dim, Residual(Attention(dim, polynomial=polynomial, heads = heads, dim_head = dim_head, cheb=cheb, K=K, alpha=alpha, beta=beta))),
                PreNorm(dim, Residual(FeedForward(dim, dropout = ff_dropout)))
            ])
            )
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)

        return x
    


class MLP(nn.Module):
    def __init__(self,dims):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2]),
            nn.Dropout(0.3)
        )
        
    def forward(self, x):
        if len(x.shape)==1:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


class simple_MLP(nn.Module):
    def __init__(self,dims):
        super(simple_MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.SiLU(),
            nn.Linear(dims[1], dims[2]),
            nn.SiLU(),
            nn.Linear(dims[2], dims[3]),
        )
        
    def forward(self, x):
        x = x.reshape(len(x), -1)
        x = self.layers(x)
        return x


class MET(nn.Module):
    def __init__(
            self, 
            small_maxlen, 
            maxlen, 
            embed_dim, 
            encoder_depth, 
            decoder_depth,
            num_heads, 
            ff_dim, 
            ff_dropout,
            enc_cheb,
            dec_cheb,
            K,
            alpha,
            beta,
            polynomial
    ):
        super().__init__()
        self.mask_emb = nn.Linear(small_maxlen, 1)
        self.maxlen = maxlen
        self.small_maxlen = small_maxlen

        self.emb_layer = TokenAndPositionEmbedding(maxlen, embed_dim)
        self.encoder = TransformerBlock(dim=embed_dim+1, 
                                        polynomial=polynomial,
                                        depth=encoder_depth, 
                                        heads=num_heads, 
                                        ff_dim=ff_dim, 
                                        dim_head=embed_dim,
                                        ff_dropout=ff_dropout,
                                        cheb=enc_cheb, 
                                        alpha=alpha,
                                        beta=beta,
                                        K=K)
        
        self.decoder = TransformerBlock(dim=embed_dim+1, 
                                        polynomial=polynomial,
                                        depth=decoder_depth, 
                                        heads=num_heads, 
                                        ff_dim=ff_dim, 
                                        dim_head=embed_dim,
                                        ff_dropout=ff_dropout,
                                        cheb=dec_cheb,
                                        alpha=alpha,
                                        beta=beta,
                                        K=K
                                        )


        self.mlp = nn.Linear(embed_dim+1, 1, bias = True)


    def forward(self, inputs, inputs_unmask_idx, inputs_mask_idx, ones):
        non_mask_embed, mask_pos = self.emb_layer(inputs, inputs_unmask_idx, inputs_mask_idx)
        non_mask_embed = self.encoder(non_mask_embed)

        mask_embed = self.mask_emb(ones)

        mask_embed = torch.unsqueeze(mask_embed, dim=1)
        mask_embed = torch.repeat_interleave(mask_embed, self.maxlen - self.small_maxlen, dim=1)
        
        mask_embed = torch.concat([mask_embed, mask_pos], dim=2)
        mask_embed = torch.concat([non_mask_embed, mask_embed], dim=1)

        h = self.decoder(mask_embed)
        prediction = self.mlp(h).squeeze()

        return prediction



def train_METModel(
    dataset_name='adult',
    embed_dim=128,
    num_heads=2,
    ff_dim=128,
    encoder_depth=6,
    decoder_depth=1,
    mask_pct=15,
    batch_size=256,
    radius=6,
    lr_adv=1e-03,
    adv_steps=5,
    save_path='./saved_models_cheb_only',
    enc_cheb=False,
    dec_cheb=False,
    K=5,
    alpha=1.0,
    beta=0.3,
    polynomial='chebyshev'):

    device = 'cuda'
    if dataset_name in ["superconductivity", "medicalcost"]:
        y_method = 'raw'
    else: y_method = 'label'
    dataset = TabularDataset(dataset=dataset_name.lower(), cont_method='minmax', categ_method='label', y_method=y_method)
    train, val, test = dataset.get_datas()
    train, val, test = pd.DataFrame(train), pd.DataFrame(val), pd.DataFrame(test)

    trn_dg = make_data(train, mask_pct)
    tst_dg = make_data(val, mask_pct)
    trn_dg = trn_dg.to(device)
    
    indexes_trn = [i for i in range(len(train.iloc[0:]))]
    indexes_tst = [i for i in range(len(val.iloc[0:]))]
    print(len(train.iloc[0]), len(val.iloc[0]))
    batch_size = min(batch_size, len(indexes_trn), len(indexes_tst))


    trainloader = DataLoader(trn_dg, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(tst_dg, batch_size=batch_size, shuffle=True)

    maxlen = train.shape[-1]-1
    small_maxlen = ((100 - mask_pct) * maxlen) // 100

    model = MET(
                small_maxlen,
                maxlen, 
                embed_dim, 
                encoder_depth, 
                decoder_depth, 
                num_heads, 
                ff_dim, 
                ff_dropout=0.2,
                enc_cheb=enc_cheb, 
                dec_cheb=dec_cheb,
                K=K,
                alpha=alpha,
                beta=beta,
                polynomial=polynomial
                ).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)

    save_path = os.path.join(save_path, dataset_name)
    os.makedirs(save_path, exist_ok=True)

    if FLAGS.model_path == '':
        checkpoint_filepath = os.path.join(save_path, f"{embed_dim}_{num_heads}_{ff_dim}_{encoder_depth}_{decoder_depth}_{FLAGS.lr}_{radius}_{K}_{FLAGS.clf_lr}_{FLAGS.polynomial}_{FLAGS.alpha}_{FLAGS.beta}")

    else:
        checkpoint_filepath = FLAGS.model_path
    print(checkpoint_filepath)

    if os.path.exists(checkpoint_filepath):
        model.load_state_dict(torch.load(checkpoint_filepath))
        print("trained model exists")
        return model
    else:
        print("trained model does not exists")
        pass 

    print(f"The number of model parameters: {num_params}")
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr) # , weight_decay=1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=0.0001) # 2

    radius = FLAGS.radius
    active_log = FLAGS.active_log
    epochs = 1000
    warmup_epochs = 1

    early_stopping = EarlyStopping(patience = 50, verbose = True, path=checkpoint_filepath)
    dim = 15
    norm_factor = torch.tensor((dim**0.5)/2, device=device, requires_grad=True)
    adv_steps = FLAGS.adv_steps
    model.train()
    for epoch in range(epochs):
        avg_loss = 0
        model.train()
        for step, batch in enumerate(trainloader):
            batch = batch.to(device)
            idx = np.cumsum([0, small_maxlen, small_maxlen, maxlen-small_maxlen, small_maxlen, maxlen])
            X = [batch[:, idx[i]:idx[i+1]].to(device) for i in range(len(idx)-1)]
            unmask, unmask_idx, mask_idx, ones, y = X[0], X[1], X[2], X[3], X[4]
            
            mu, sigma = torch.zeros(unmask.shape, device=device), torch.ones(unmask.shape, device=device)  # mean and standard deviation
            unmask_idx = unmask_idx.to(torch.long)
            mask_idx = mask_idx.to(torch.long)
            ones = ones.to(device).float()
            y = y.float()

            mu.requires_grad = True
            sigma.requires_grad = True

            if epoch >= warmup_epochs and step % 2 == 1:
                x_adv_0 = torch.tensor(unmask + (torch.normal(mu, sigma)) / norm_factor, dtype=torch.float, requires_grad=True, device=device)
                for j in range(adv_steps):
                    model.eval()
                    optimizer.zero_grad()
    
                    x_adv_0 = torch.tensor(x_adv_0, dtype=torch.float, requires_grad=True, device=device)
                    y_pred = model(x_adv_0, unmask_idx, mask_idx, ones)  # Forward pass
                    loss_adv = loss_fn(y, y_pred)
                    
                    gradient = torch.autograd.grad(loss_adv, x_adv_0, retain_graph=True)
                    gradient_norm = gradient[0] / torch.norm(gradient[0])
                    x_adv_0 = x_adv_0 + lr_adv * gradient_norm 
                    h = x_adv_0 - unmask 
                    norm_h = torch.norm(h)
                    clip_norm_h = torch.clamp(norm_h, max=radius)
                    new_h = h / norm_h * clip_norm_h
                    x_adv_0 = unmask + new_h
                    
                model.train()

                y_pred = model(unmask, unmask_idx, mask_idx, ones)  # Forward pass
                y_pred_adv = model(x_adv_0, unmask_idx, mask_idx, ones)  # Forward pass
                loss = loss_fn(torch.cat([y, y], axis=0), torch.cat([y_pred_adv, y_pred] , axis=0))

                loss.backward()
                optimizer.step()
                scheduler.step()
                avg_loss = avg_loss + loss
                if active_log:
                    wandb.log({"step" : step, "Adversarial Train Loss" : avg_loss/len(trainloader)})

            else:
                optimizer.zero_grad()

                y_pred = model(unmask, unmask_idx, mask_idx, ones)  # Forward pass
                loss = loss_fn(y, y_pred)
                loss.backward()
                optimizer.step()
                scheduler.step()

                avg_loss = avg_loss + loss
        
        print(f"Epoch: {epoch} -- Train Loss: {avg_loss/len(trainloader)}")
        if active_log:
            wandb.log({"Epoch" : epoch, "Avg. Train Loss" : avg_loss/len(trainloader)})

        val_avg_loss = 0
        model.eval()

        for step, batch in enumerate(testloader):
            batch = batch.to(device)
            idx = np.cumsum([0, small_maxlen, small_maxlen, maxlen-small_maxlen, small_maxlen, maxlen])
            X = [batch[:, idx[i]:idx[i+1]].to(device) for i in range(len(idx)-1)]
            unmask, unmask_idx, mask_idx, ones, y = X[0], X[1], X[2], X[3], X[4]

            unmask_idx = unmask_idx.to(torch.long)
            mask_idx = mask_idx.to(torch.long)
            ones = ones.to(device).float()
            y = y.float()

            y_pred = model(unmask, unmask_idx, mask_idx, ones)  # Forward pass
            loss = loss_fn(y, y_pred)

            val_avg_loss = avg_loss + loss

        early_stopping(val_avg_loss/len(testloader), model)
        if early_stopping.early_stop:
            break


        print(f"Epoch: {epoch} -- Val Loss: {val_avg_loss/len(testloader)}")
        if active_log:
            wandb.log({"Epoch":epoch,"Avg. Val Loss" : val_avg_loss/len(testloader)})
    torch.save(model.state_dict(), checkpoint_filepath)
    return model



def train_and_eval_met(dataset_name, model,
                       embed_dim=128, batch_size=40, save_path='./saved_models_cheb_only'):
    
    save_path = os.path.join(save_path, dataset_name)
    os.makedirs(save_path, exist_ok=True)

    if FLAGS.model_path_linear == '':
        checkpoint_filepath = f"{save_path}/{embed_dim}_{FLAGS.num_heads}_{FLAGS.ff_dim}_{FLAGS.encoder_depth}_{FLAGS.decoder_depth}_{FLAGS.lr}_{FLAGS.radius}_{FLAGS.K}_{FLAGS.clf_lr}_{FLAGS.polynomial}_{FLAGS.alpha}_{FLAGS.beta}_ae"
    else:
        checkpoint_filepath = FLAGS.model_path_linear
    
    device = 'cuda'
    if dataset_name in ['medicalcost', "superconductivity"]:
        y_method = 'raw'
    else: y_method = 'label'
    dataset = TabularDataset(dataset=dataset_name.lower(), cont_method='raw', categ_method='label', y_method=y_method)
    train, val, test = dataset.get_datas()

    train_y, val_y, test_y = train[:, -1:], val[:, -1:], test[:, -1:]
    classification = False if len(np.unique(train_y)) > 10 else True
    if classification:
        ohe = OneHotEncoder(sparse=False)
        train_y = ohe.fit_transform(train_y.reshape(-1, 1))
        val_y = ohe.transform(val_y.reshape(-1, 1))
        test_y = ohe.transform(test_y.reshape(-1, 1))

        num_class = train_y.shape[-1]
        train, val, test = train[:, :-num_class], val[:, :-num_class], test[:, :-num_class]
    else:
        train, val, test = train[:, :-1], val[:, :-1], test[:, :-1]
        num_class = 1
    
    input_dim = train.shape[-1]

    indexes_trn = [i for i in range(len(train))]
    indexes_val = [i for i in range(len(val))]
    indexes_tst = [i for i in range(len(test))]
    batch_size = min(batch_size, len(indexes_trn), len(indexes_tst))

    trainloader = DataLoader(np.concatenate([train, train_y], axis=1), batch_size=batch_size, shuffle=True)
    valloader = DataLoader(np.concatenate([val, val_y], axis=1), batch_size=batch_size, shuffle=True)
    testloader = DataLoader(np.concatenate([test, test_y], axis=1), batch_size=batch_size, shuffle=True)

    mlp_model = simple_MLP([(embed_dim+1) * input_dim, 16, 8, num_class]).to(device)

    num_params = sum(p.numel() for p in mlp_model.parameters() if p.requires_grad)
    print(mlp_model)
    print(f"The number of mlp_model parameters: {num_params}")


    train_flag = True
    if classification:
        loss_fn = nn.CrossEntropyLoss()
        metric = roc_auc_score
    else:
        loss_fn = nn.MSELoss()
        metric= r2_score

    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=FLAGS.clf_lr) 
    vae_optimizer = torch.optim.Adam(model.encoder.parameters(), lr=FLAGS.clf_lr) 

    if train_flag:
        model.train()
        model.to(device)
        best_val_auc = 0
        best_val_r2 = 0
        early_stopping = EarlyStopping(patience = 30, verbose = True)

        for epoch in range(1000):
            avg_loss = 0
            avg_acc = 0
            avg_auc = 0
            avg_r2 = 0

            mlp_model.train()
            model.train()
            iter = 0
            for iter, batch in enumerate(trainloader):
                iter += 1 
                optimizer.zero_grad()
                vae_optimizer.zero_grad()
                X, y = batch[:, :input_dim].to(device), batch[:, input_dim:].to(device)
                r, c = X.shape
                y = y.to(torch.float32)
                unmask_pos = torch.tensor([np.arange(0, c, 1) for _ in range(r)], device=device, dtype=torch.long)
                mask_pos = torch.tensor([[] for _ in range(r)], device=device, dtype=torch.long)

                embed, _ = model.emb_layer(X, unmask_pos, mask_pos)
                representation = model.encoder(embed)

                pred = mlp_model(representation)

                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                vae_optimizer.step()

                if classification:
                    train_acc = accuracy_score(y.cpu().detach().argmax(axis=1), pred.cpu().detach().argmax(axis=1))
                    try:
                        train_auc = roc_auc_score( y.cpu().detach()[:, 1], pred.cpu().detach()[:, 1])
                    except:
                        train_auc=0.5
                else:
                    train_r2 = r2_score(y.cpu().detach(), pred.cpu().detach())

                avg_loss += loss
                if classification:
                    avg_acc += train_acc
                    avg_auc += train_auc
                    print(f"EPOCH {epoch} | train loss: {avg_loss/(iter)} | Acc: {avg_acc/(iter)} | AUC: {avg_auc/(iter)}")
                    if FLAGS.active_log:
                        wandb.log({"Epoch":epoch,"Avg. train Loss" : avg_loss/(iter), "train acc":avg_acc/(iter), "train auc":avg_auc/(iter)})
                else:
                    avg_r2 += train_r2
                    print(f"EPOCH {epoch} | train loss: {avg_loss/(iter)} | R2: {avg_r2/(iter)}")
                    if FLAGS.active_log:
                        wandb.log({"Epoch":epoch,"Avg. train Loss" : avg_loss/(iter), "train R2":avg_r2/(iter)})

            avg_loss = 0
            avg_acc = 0
            avg_auc = 0
            avg_r2 = 0

            mlp_model.eval()
            model.eval()
            iter = 0
            for iter, batch in enumerate(valloader):
                iter += 1
                batch.to(device)
                X, y = batch[:, :input_dim].to(device), batch[:, input_dim:].to(device)
                r, c = X.shape
                unmask_pos = torch.tensor([np.arange(0, c, 1) for _ in range(r)], device=device, dtype=torch.long)
                mask_pos = torch.tensor([[] for _ in range(r)], device=device, dtype=torch.long)

                embed, _ = model.emb_layer(X, unmask_pos, mask_pos)
                representation = model.encoder(embed)
                pred = mlp_model(representation)
                loss = loss_fn(pred, y)

                if classification:
                    val_acc = accuracy_score(y.cpu().detach().argmax(axis=1), pred.cpu().detach().argmax(axis=1))
                    try:
                        val_auc = roc_auc_score( y.cpu().detach()[:, 1], pred.cpu().detach()[:, 1])
                    except:
                        val_auc=0.5
                else:
                    val_r2 = r2_score(y.cpu().detach(), pred.cpu().detach())

                avg_loss += loss
                if classification:
                    avg_acc += val_acc
                    avg_auc += val_auc
                else:
                    avg_r2 += val_r2

            if classification:
                early_stopping(-avg_auc/(iter), model)
            else:
                early_stopping(-avg_r2/(iter), model)

            if early_stopping.early_stop: 
                break
            if classification:
                print(f"EPOCH {epoch} | val loss: {avg_loss/(iter)} | Acc: {avg_acc/(iter)} | AUC:  {avg_auc/(iter)}")
                if FLAGS.active_log:
                    wandb.log({"Epoch":epoch,"Avg. val Loss" : avg_loss/(iter), "val acc":avg_acc/(iter), "val auc":avg_auc/(iter)})

                if (avg_auc / (iter)) >= best_val_auc:
                    best_val_auc = avg_auc / (iter)
                    torch.save(mlp_model.state_dict(), checkpoint_filepath)
                    torch.save(model.state_dict(), f"{checkpoint_filepath}_encoder")
                    print(f"Model saved at {checkpoint_filepath}, Avg. acc: {avg_acc / (iter)}, Avg. auc: {avg_auc / (iter)} ")
            else:
                print(f"EPOCH {epoch} | val loss: {avg_loss/(iter)} | R2: {avg_r2/(iter)}")
                if FLAGS.active_log:
                    wandb.log({"Epoch":epoch,"Avg. val Loss" : avg_loss/(iter), "val r2":avg_acc/(iter)})

                if (avg_r2 / (iter)) >= best_val_r2:
                    best_val_r2 = avg_r2 / (iter)
                    torch.save(mlp_model.state_dict(), checkpoint_filepath)
                    torch.save(model.state_dict(), f"{checkpoint_filepath}_encoder")
                    print(f"Model saved at {checkpoint_filepath}, Avg. r2: {avg_r2 / (iter)} ")

    if not os.path.exists(checkpoint_filepath):
        torch.save(mlp_model.state_dict(), checkpoint_filepath)
       
    mlp_model.eval()
    model.eval()
    avg_acc = 0 
    avg_auc = 0
    avg_loss = 0
    avg_r2 = 0

    model.load_state_dict(torch.load(f"{checkpoint_filepath}_encoder"))
    mlp_model.load_state_dict(torch.load(checkpoint_filepath))
    iter = 0
    with torch.no_grad():
        test_data = torch.tensor(np.concatenate([test, test_y], axis=1)).to(device)
        X, y = test_data[:, :input_dim].to(device), test_data[:, input_dim:].to(device)
        r, c = X.shape
        unmask_pos = torch.tensor([np.arange(0, c, 1) for _ in range(r)], device=device, dtype=torch.long)
        mask_pos = torch.tensor([[] for _ in range(r)], device=device, dtype=torch.long)
        embed, _ = model.emb_layer(X, unmask_pos, mask_pos)
        representation = model.encoder(embed)
        pred = mlp_model(representation)

        if classification:
            test_acc = accuracy_score(y.cpu().detach().argmax(axis=1), pred.cpu().detach().argmax(axis=1))
            test_auc = roc_auc_score( y.cpu().detach()[:, 1], pred.cpu().detach()[:, 1])
            avg_acc += test_acc
            avg_auc += test_auc
        else:
            test_r2 = r2_score(y.cpu().detach(), pred.cpu().detach())
            avg_r2 += test_r2

    if classification:
        test_acc =  avg_acc
        test_auc =  avg_auc
        if FLAGS.active_log:
            wandb.log({"Test acc" :test_acc, "test auc":test_auc})
        return test_auc
    else:
        test_r2 =  avg_r2
        if FLAGS.active_log:
            wandb.log({"Test r2" :test_r2})
        return test_r2




def main(argv: Sequence[str]) -> None:
    name = f"{FLAGS.embed_dim}_{FLAGS.num_heads}_{FLAGS.ff_dim}_{FLAGS.encoder_depth}_{FLAGS.decoder_depth}_{FLAGS.lr}_{FLAGS.radius}_{FLAGS.K}_{FLAGS.clf_lr}_{FLAGS.polynomial}_{FLAGS.alpha}_{FLAGS.beta}"

    if FLAGS.active_log:
        wandb.init(project = "met_tabpsa", group = FLAGS.dataset_name ,name = name)
    params = [{
        "dataset_name": FLAGS.dataset_name,
        "embed_dim": FLAGS.embed_dim,
        "num_heads": FLAGS.num_heads,
        "ff_dim": FLAGS.ff_dim, 
        "encoder_depth": FLAGS.encoder_depth, 
        "decoder_depth": FLAGS.decoder_depth, 
        "mask_pct": FLAGS.mask_pct, 
        "lr": FLAGS.lr, 
        "lr_adv": FLAGS.lr_adv, 
        "clf_lr": FLAGS.clf_lr, 
        "enc_cheb": FLAGS.enc_cheb, 
        "dec_cheb": FLAGS.dec_cheb,
        "K": FLAGS.K,
        "alpha": FLAGS.alpha,
        "beta": FLAGS.beta,
        "polynomial": FLAGS.polynomial
    }]
    params = pd.DataFrame(params)
    print(params)
    
    # final_score = []
    # for i in range(5):
    met_model = train_METModel(
        dataset_name=FLAGS.dataset_name,
        embed_dim=FLAGS.embed_dim,
        num_heads=FLAGS.num_heads,
        ff_dim=FLAGS.ff_dim,
        encoder_depth=FLAGS.encoder_depth,
        decoder_depth=FLAGS.decoder_depth,
        mask_pct=FLAGS.mask_pct,
        radius=FLAGS.radius,
        lr_adv=FLAGS.lr_adv,
        adv_steps=FLAGS.adv_steps,
        batch_size=256, 
        enc_cheb=FLAGS.enc_cheb,
        dec_cheb=FLAGS.dec_cheb,
        K=FLAGS.K,
        alpha=FLAGS.alpha,
        beta=FLAGS.beta,
        save_path=FLAGS.save_path,
        polynomial=FLAGS.polynomial)
    met_auc = train_and_eval_met(FLAGS.dataset_name,met_model,FLAGS.embed_dim,batch_size=256, save_path=FLAGS.save_path)

    params["auc"] = met_auc
    print(f"final auroc: {met_auc}")
    if FLAGS.active_log:
        wandb.log({"final auroc" :met_auc})

    print(params)
    if not os.path.isfile('./result.csv'):
        params.to_csv('./result.csv', header='column_names')
        params.to_csv('./result.csv', mode='a', header=False)

if __name__ == '__main__':
  app.run(main)