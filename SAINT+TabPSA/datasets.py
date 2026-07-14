import os
import torch
import numpy  as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from scipy.io import arff

np.random.seed(0)

FOLDER_PATH = '../'

class TabularDataset():
    def __init__(self, dataset, cont_method='minmax', categ_method='label', y_method='label'):
        """
        Arguments
        - dataset: dataset name
        - cont_method: [ minmax, label, onehot, raw ]
        - categ_method: [ minmax, label, onehot, raw ]
        - y_method: [ minmax, label, onehot, raw ]

        Notes
        - If the 'method' is set to 'label,' it is possible for a 'DataConversionWarning' to be raised, but this is not a cause for concern.
        """

        load_data = {
            "income": self.process_income,
            "default": self.process_default,
            "phishing": self.process_phishing,
            "alphabank": self.process_alphabank, 
            "clave": self.process_clave,
            "contraceptive": self.process_contraceptive,
            "activity": self.process_activity,
            "buddy": self.process_buddy, 
            "medicalcost": self.process_medicalcost,
            "superconductivity": self.process_superconductivity, 
        }
        assert dataset in load_data.keys()

        preprocess_method = {
            "minmax": MinMaxScaler(), 
            "label": LabelEncoder(),
            # "onehot": OneHotEncoder(sparse_output=False),
            "onehot": OneHotEncoder(),
            "standard": StandardScaler(),
            "raw": None
        }
        assert cont_method.lower() in preprocess_method.keys()
        assert categ_method.lower() in preprocess_method.keys()
        assert y_method.lower() in preprocess_method.keys()

        # get data and index of categorical/continuous
        self.train, self.val, self.test, self.categ_index, self.cont_index = load_data[dataset]()
        
        # index
        last_index = self.train.shape[-1]-1
        self.y_index = [ last_index ]
        if last_index in self.categ_index:
            self.categ_index.remove(last_index)
        if last_index in self.cont_index:
            self.cont_index.remove(last_index)

        # get encoder (or scaler)
        self.cont_encoder = preprocess_method[cont_method.lower()]
        self.categ_encoder = preprocess_method[categ_method.lower()]
        self.y_encoder = preprocess_method[y_method.lower()]
        self.categ_method = categ_method.lower()

        # raw column
        if cont_method.lower() == 'raw': self.cont_index = []
        if categ_method.lower() == 'raw': self.categ_index = []
        if y_method.lower() == 'raw': self.y_index = []
        
        # preprocessing
        self.train_x, self.train_y, self.val_x, self.val_y, \
            self.test_x, self.test_y, self.categ_dims = self.preprocessing()

    def get_datas(self, seperate_y=False):
        if seperate_y:
            return self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y
        else:
            return ( np.concatenate([self.train_x, self.train_y], axis=1).astype(float), 
                     np.concatenate([self.val_x, self.val_y], axis=1).astype(float),
                     np.concatenate([self.test_x, self.test_y], axis=1).astype(float) )
    
    def get_index(self):
        return self.categ_index, self.cont_index
    
    def get_categ_dims(self):
        return self.categ_dims

    def get_infos(self):
        print('dataset size:', self.train_x.shape[0]+self.val_x.shape[0]+self.test_x.shape[0])
        print('train shape:', self.train_x.shape)
        print('val shape:', self.val_x.shape)
        print('test shape:', self.test_x.shape)
        print()
        print('number of features:', self.train_x.shape[-1])
        print('number of continuous:', len(self.cont_index))
        print('number of categorical:', len(self.categ_index))


    def preprocessing(self):
        train_encoded_data = []
        val_encoded_data = []
        test_encoded_data = []

        categ_dims = []
        for i in range(self.train.shape[-1]):
            train_curr = self.train[:, i].copy().reshape(-1, 1)
            val_curr = self.val[:, i].copy().reshape(-1, 1)
            test_curr = self.test[:, i].copy().reshape(-1, 1)

            if i in self.y_index:
                train_encoded = self.y_encoder.fit_transform(train_curr)
                val_encoded = self.y_encoder.transform(val_curr)
                test_encoded = self.y_encoder.transform(test_curr)
            
            elif i in self.categ_index:
                train_encoded = self.categ_encoder.fit_transform(train_curr)
                val_encoded = self.categ_encoder.transform(val_curr)
                test_encoded = self.categ_encoder.transform(test_curr)            
                try:
                    categ_dims.append(len(self.categ_encoder.classes_))
                except:
                    categ_dims.append(len(self.categ_encoder.categories_))

            elif i in self.cont_index:
                train_encoded = self.cont_encoder.fit_transform(train_curr)
                val_encoded = self.cont_encoder.transform(val_curr)
                test_encoded = self.cont_encoder.transform(test_curr)  

            else: # raw
                train_encoded = train_curr
                val_encoded = val_curr
                test_encoded = test_curr

            train_encoded_data.append(train_encoded.reshape(len(self.train), -1))
            val_encoded_data.append(val_encoded.reshape(len(self.val), -1))
            test_encoded_data.append(test_encoded.reshape(len(self.test), -1))
        
        train = np.concatenate(train_encoded_data, axis=1)
        val = np.concatenate(val_encoded_data, axis=1)
        test = np.concatenate(test_encoded_data, axis=1)

        train_x, train_y = train[:, :-1], train[:, -1:]
        val_x, val_y = val[:, :-1], val[:, -1:]
        test_x, test_y = test[:, :-1], test[:, -1:]

        return train_x, train_y, val_x, val_y, test_x, test_y, categ_dims

    def process_income(self):
        RANDOMSEED = 1
        train = pd.read_csv(os.path.join(FOLDER_PATH,"data/income_train.csv"))
        test = pd.read_csv(os.path.join(FOLDER_PATH, "data/income_test.csv"))

        train = np.array(train.dropna(axis=0))
        test = np.array(test.dropna(axis=0))

        train, val = train[int(len(test)/2):], train[:int(len(test)/2)]

        categorical = [1, 3, 5, 6, 7, 8, 9, 13, 14]
        continuous = list(set(list(range(train.shape[-1]))[:-1]) - set(categorical))
        
        return train, val, test, categorical, continuous

     
    def process_default(self): # binary
        RANDOMSEED = 3
        data = pd.read_csv(os.path.join(FOLDER_PATH, "data/default.csv"), header=1, index_col=0)
        data = np.array(data.dropna(axis=0))

        train, test = train_test_split(data, test_size=0.2, shuffle=True, random_state=RANDOMSEED) 
        train = np.array(train)
        test = np.array(test)
        train, val = train[int(len(test)/2):], train[:int(len(test)/2)]

        continuous = [0, 2, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        categorical = list(set(list(range(data.shape[-1]))) - set(continuous))

        return train, val, test, categorical, continuous


    def process_phishing(self): # binary
        RANDOMSEED = 1

        data = pd.read_csv(os.path.join(FOLDER_PATH, "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"), header=0)
        data = np.array(data.dropna(axis=0))[:, 1:]
        data = data[data[:, -2] != ' ']

        train, test = train_test_split(data, test_size=0.15, shuffle=True, random_state=RANDOMSEED) 
        train = np.array(train)
        test = np.array(test)
        train, val = train[int(len(test)/2):], train[:int(len(test)/2)]

        continuous = [4, 17, 18]
        categorical = list(set(list(range(data.shape[-1]))) - set(continuous))

        return train, val, test, categorical, continuous

    
    def process_alphabank(self):
        RANDOMSEED = 1
        data = pd.read_csv(os.path.join(FOLDER_PATH, "data/Alpha_bank.csv"))

        train, test = train_test_split(data, test_size=0.3, shuffle=True, random_state=RANDOMSEED) 
        test, val = train_test_split(test, test_size=0.5, shuffle=True, random_state=RANDOMSEED) 

        train = np.array(train.dropna(axis=0))
        val = np.array(val.dropna(axis=0))
        test = np.array(test.dropna(axis=0))

        continuous = [0]
        categorical = list(set(list(range(data.shape[-1]))[:-1]) - set(continuous))

        return train, val, test, categorical, continuous

    
    def process_clave(self):
        RANDOMSEED = 1

        file_path = os.path.join(FOLDER_PATH, "data/ClaveVectors_Firm-Teacher_Model.txt")
        f = open(file_path, 'r')

        data = []
        while True:
            line = f.readline()
            if not line: break
            data.append(line.split()[:20])

        data = np.array(data).astype('int64')

        data_x = data[:, :-4]
        data_y = data[:, -4:].argmax(axis=1).reshape(-1, 1)
        data = np.concatenate([data_x, data_y], axis=1)

        train, test = train_test_split(data, test_size=0.3, shuffle=True, random_state=RANDOMSEED) 
        test, val = train_test_split(test, test_size=0.5, shuffle=True, random_state=RANDOMSEED) 

        continuous = []
        categorical = list(set(list(range(train.shape[-1]))[:-1]) - set(continuous))

        return train, val, test, categorical, continuous

    def process_contraceptive(self):
        RANDOMSEED = 1
        data = pd.read_csv(os.path.join(FOLDER_PATH, "data/contraceptive+method+choice/cmc.data"), header=None)

        train, test = train_test_split(data, test_size=0.3, shuffle=True, random_state=RANDOMSEED) 
        test, val = train_test_split(test, test_size=0.5, shuffle=True, random_state=RANDOMSEED) 

        train = np.array(train)
        val = np.array(val)
        test = np.array(test)

        continuous = [0, 3]
        categorical = list(set(list(range(data.shape[-1]))[:-1]) - set(continuous))

        return train, val, test, categorical, continuous


    def process_activity(self):
        RANDOMSEED = 1
        
        data = pd.read_csv(os.path.join(FOLDER_PATH, "data/aw_fb_data.csv"), index_col=0)
        data = data.drop('X1', axis=1)
        train, test = train_test_split(data, test_size=0.3, shuffle=True, random_state=RANDOMSEED) 
        test, val = train_test_split(test, test_size=0.45, shuffle=True, random_state=RANDOMSEED) 

        train = np.array(train)
        val = np.array(val)
        test = np.array(test)

        categorical = [1, 16]
        continuous = list(set(list(range(data.shape[-1]))[:-1]) - set(categorical))

        return train, val, test, categorical, continuous


    def process_buddy(self):
        RANDOMSEED = 1

        data = pd.read_csv(os.path.join(FOLDER_PATH, "data/buddy_train.csv")) # test dataset doesn't have target feature
        data = data.dropna(axis=0).iloc[:, 1:]

        data['issue_date'] = pd.to_datetime(data['issue_date']).astype(int)
        data['listing_date'] = pd.to_datetime(data['listing_date']).astype(int)
            
        train, test = train_test_split(data, test_size=0.3, shuffle=True, random_state=RANDOMSEED) 
        test, val = train_test_split(test, test_size=0.4, shuffle=True, random_state=RANDOMSEED) 

        train = np.array(train)
        val = np.array(val)
        test = np.array(test)

        categorical = [2, 3, 8]
        continuous = list(set(list(range(train.shape[-1]))[:-1]) - set(categorical))

        return train, val, test, categorical, continuous


    def process_medicalcost(self):
        RANDOMSEED = 1

        data = pd.read_csv(os.path.join(FOLDER_PATH, "data/insurance.csv"))

        train, test = train_test_split(data, test_size=0.25, shuffle=True, random_state=RANDOMSEED) 
        test, val = train_test_split(test, test_size=0.4, shuffle=True, random_state=RANDOMSEED) 

        train = np.array(train)
        val = np.array(val)
        test = np.array(test)

        categorical = [1, 4, 5]
        continuous = list(set(list(range(train.shape[-1]))[:-1]) - set(categorical))

        return train, val, test, categorical, continuous


    def process_superconductivity(self):
        RANDOMSEED = 1

        data = pd.read_csv(os.path.join(FOLDER_PATH, "data/superconductivty+data/train.csv"))

        train, test = train_test_split(data, test_size=0.3, shuffle=True, random_state=RANDOMSEED) 
        test, val = train_test_split(test, test_size=0.4, shuffle=True, random_state=RANDOMSEED) 

        train = np.array(train)
        val = np.array(val)
        test = np.array(test)

        categorical = []
        continuous = list(set(list(range(train.shape[-1]))[:-1]) - set(categorical))

        return train, val, test, categorical, continuous
