import numpy  as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataset import TabularDataset
np.random.seed(0)

def get_dataloader(args, dataset):

    if args.task != 'regression':
        tabular_dataset = TabularDataset(dataset, cont_method='minmax', categ_method='label', y_method='label')
    else:
        tabular_dataset = TabularDataset(dataset, cont_method='minmax', categ_method='label', y_method='raw')

    x_train, y_train, x_valid, y_valid, x_test, y_test = tabular_dataset.get_datas(True)
    categ_index, cont_index = tabular_dataset.get_index()    

    train_dataset = TabTransformerDataset(x_train, y_train, categ_index, cont_index)
    val_dataset = TabTransformerDataset(x_valid, y_valid, categ_index, cont_index)
    test_dataset = TabTransformerDataset(x_test, y_test, categ_index, cont_index)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    return train_dataloader, val_dataloader, test_dataloader, tabular_dataset

class TabTransformerDataset(Dataset):
    def __init__(self, x, y, categorical_index, continuous_index):
        self.x_categ = x[:, categorical_index].astype('float32')
        self.x_cont = x[:, continuous_index].astype('float32')
        self.y = y.astype('float64')

        self.mask_ratio = 0.3 # in paper
        self.num_categ = len(categorical_index)
        self.num_cont = len(continuous_index)
        self.num_categ_mask = int( np.ceil( self.num_categ * self.mask_ratio ) )
        self.num_cont_mask = int( np.ceil( self.num_cont * self.mask_ratio ) )

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):

        # masking categorical
        categ_mask_index = np.sort(np.random.choice(self.num_categ, size=self.num_categ_mask, replace=False))
        x_categ_masked = self.x_categ.copy()
        x_categ_masked = x_categ_masked[index] + 1 # 0 index for masking
        x_categ_masked[categ_mask_index] = 0

        # masking continuous
        cont_mask_index = np.sort(np.random.choice(self.num_cont, size=self.num_cont_mask, replace=False))
        x_cont_masked = self.x_cont.copy()
        x_cont_masked = x_cont_masked[index]
        x_cont_masked[cont_mask_index] = 0

        return (
            torch.tensor(self.x_categ[index]), 
            torch.tensor(self.x_cont[index]), 
            torch.tensor(self.y[index]),
            torch.tensor(x_categ_masked),
            torch.tensor(x_cont_masked),
            torch.tensor(categ_mask_index),
            torch.tensor(cont_mask_index),
        )
