import os
import math
import random
import torch
import datetime
import argparse
import numpy as np
import logging

def basic_setting():
    
    # argument
    args = parse_args()
    args.log_path = os.path.join(args.output_dir, args.log_name+'.log')
    args.pretrain_checkpoint_path = os.path.join(args.output_dir, args.log_name+'_pretrain.pt')
    args.finetune_checkpoint_path = os.path.join(args.output_dir, args.log_name+'_finetune.pt')
    args.mlp_checkpoint_path = os.path.join(args.output_dir, args.log_name+'_mlp.pt')
    
    # logger setting
    logger = set_logger(args.log_path)

    # gpu setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    return args, logger

def set_logger(log_path, log_name='tabtrans', mode='a'):
    """set up log file
    mode : 'a'/'w' mean append/overwrite,
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_path, mode=mode)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False  # prevent the child logger from propagating log to the root logger (twice), not necessary
    return logger

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created')

def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S-%f')

    return cur

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="income", type=str)
    parser.add_argument("--task", default='binary', type=str)

    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--log_name", default=get_local_time(), type=str)

    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--gpu_id", default="0", type=str, help="gpu_id")

    parser.add_argument("--epoch", default=10000, type=int, help="number of epochs")
    parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight_decay of adam")
    parser.add_argument("--learning_rate", default=5e-3, type=float, help="learning rate of adam")
    parser.add_argument("--batch_size", default=1024, type=int, help="number of batch_size")
    parser.add_argument('--early_stopping_rounds', default=15, type=int)

    parser.add_argument('--K', default=3, type=int)
    parser.add_argument("--polynomial", default="chebyshev", type=str, help="polynomial type")
    parser.add_argument("--dim", default=32, type=int)
    parser.add_argument("--depth", default=6, type=int)
    parser.add_argument("--n_heads", default=4, type=int)
    parser.add_argument("--hidden_mlp", default=512, type=int)
    parser.add_argument("--attn_dropout", default=0.1, type=float)
    parser.add_argument("--ff_dropout", default=0.1, type=float)
    parser.add_argument("--categ_loss_ratio", default=0.1, type=float)

    return parser.parse_args()

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, checkpoint_path, logger, patience=10, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.logger = logger

    def compare(self, score):
        for i in range(len(score)):
            if score[i] > self.best_score[i]+self.delta:
                return False
        return True

    def __call__(self, score, model, multiple_models=False):
        # score HIT@10 NDCG@10

        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0]*len(score))
            self.save_checkpoint(score, model, multiple_models)
        elif self.compare(score):
            self.counter += 1
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model, multiple_models)
            self.counter = 0

    def save_checkpoint(self, score, model, multiple_models):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.logger.info(f'Validation score increased.  Saving model ...')
        
        if multiple_models:
            assert len(model) == len(self.checkpoint_path)
            for i in range(len(model)):
                torch.save(model[i].state_dict(), self.checkpoint_path[i])
            self.score_min = score
        else:
            torch.save(model.state_dict(), self.checkpoint_path)
            self.score_min = score
