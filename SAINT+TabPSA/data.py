# import openml
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch.utils.data import Dataset
from datasets import TabularDataset

def simple_lapsed_time(text, lapsed):
    hours, rem = divmod(lapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(text+": {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


def concat_data(X,y):
    # import ipdb; ipdb.set_trace()
    return pd.concat([pd.DataFrame(X['data']), pd.DataFrame(y['data'][:,0].tolist(),columns=['target'])], axis=1)


def data_split(X,y,nan_mask):
    x_d = {
        'data': X.values,
        'mask': nan_mask.values
    }
    
    if x_d['data'].shape != x_d['mask'].shape:
        raise'Shape of data not same as that of nan mask!'
        
    y_d = {
        'data': y.reshape(-1, 1)
    } 
    return x_d, y_d


def load_data(dataset_name, seed):

    np.random.seed(seed) 
    if dataset_name.lower() in ['medicalcost', 'superconductivity']:
        y_method = 'minmax'
    else: y_method = 'label'

    dataset = TabularDataset(dataset=dataset_name.lower(), cont_method='minmax', categ_method='label', y_method=y_method)
    x_train, y_train, x_valid, y_valid, x_test, y_test = dataset.get_datas(True)
    x_train, y_train, x_valid, y_valid, x_test, y_test = pd.DataFrame(x_train), pd.DataFrame(y_train), pd.DataFrame(x_valid), pd.DataFrame(y_valid), pd.DataFrame(x_test), pd.DataFrame(y_test)
    cat_idxs, con_idxs = dataset.get_index()
    cat_dims = dataset.get_categ_dims()

    if x_valid is None:
        x_valid, y_valid = x_train[:int(len(x_train)*0.1)], y_train[:int(len(y_train)*0.1)]
        x_train, y_train = x_train[int(len(x_train)*0.1):], y_train[int(len(y_train)*0.1):]

    temp_train = x_train.fillna("MissingValue")
    temp_valid = x_valid.fillna("MissingValue")
    temp_test = x_test.fillna("MissingValue")
    nan_mask_train = temp_train.ne("MissingValue").astype(int)
    nan_mask_valid = temp_valid.ne("MissingValue").astype(int)
    nan_mask_test = temp_test.ne("MissingValue").astype(int)
    
    y_train, y_valid, y_test = np.array(y_train), np.array(y_valid), np.array(y_test)
    x_train, y_train = data_split(x_train,y_train,nan_mask_train)
    x_valid, y_valid = data_split(x_valid,y_valid,nan_mask_valid)
    x_test, y_test = data_split(x_test,y_test,nan_mask_test)

    return cat_dims, cat_idxs, con_idxs, x_train, y_train, x_valid, y_valid, x_test, y_test




class DataSetCatCon(Dataset):
    def __init__(self, X, Y, cat_cols,task='clf', continuous_mean_std=None):
        
        cat_cols = list(cat_cols)
        X_mask =  X['mask'].copy()
        X = X['data'].copy()
        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        self.X1 = X[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2 = X[:,con_cols].copy().astype(np.float32) #numerical columns
        self.X1_mask = X_mask[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2_mask = X_mask[:,con_cols].copy().astype(np.int64) #numerical columns
        if task == 'clf':
            self.y = Y['data']#.astype(np.float32)
        else:
            self.y = Y['data'].astype(np.float32)
        self.cls = np.zeros_like(self.y,dtype=int)
        self.cls_mask = np.ones_like(self.y,dtype=int)

        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous
        return np.concatenate((self.cls[idx], self.X1[idx])), self.X2[idx],self.y[idx], np.concatenate((self.cls_mask[idx], self.X1_mask[idx])), self.X2_mask[idx]
