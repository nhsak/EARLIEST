import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

class SyntheticTimeSeries(Dataset):
    def __init__(self, args):
        self.nseries = args.nseries
        self.ntimesteps = args.ntimesteps
        self.data, self.labels, self.signal_locs = self.generateDataset()
        #self.train_ix, self.val_ix, self.test_ix = self.getSplitIndices()
        self.N_FEATURES = 1
        self.N_CLASSES = len(np.unique(self.labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        return self.data[ix], self.labels[ix]

    def generateDataset(self):
        self.signal_locs = np.random.randint(self.ntimesteps, size=int(self.nseries))
        X = np.zeros((self.nseries, self.ntimesteps, 1))
        y = np.zeros((self.nseries))

        for i in range(int(self.nseries)):
            if i < (int(self.nseries/2.)):
                X[i, self.signal_locs[i], 0] = 1
                y[i] = 1
            else:
                X[i, self.signal_locs[i], 0] = 0

        self.signal_locs[int(self.nseries/2):] = -1 
        data = torch.tensor(np.asarray(X).astype(np.float32),
                            dtype=torch.float)
        labels = torch.tensor(np.array(y).astype(np.int32), dtype=torch.long)
        signal_locs = torch.tensor(np.asarray(self.signal_locs),
                                   dtype=torch.float)
        print(data.shape)
        return data, labels, signal_locs


class BugSenseTimeSeries(Dataset):
    def __init__(self, args):
        self.ntimesteps = 80
        self.data, self.labels = self.load_dataset()
        #self.train_ix, self.val_ix, self.test_ix = self.getSplitIndices()
        self.N_FEATURES = 24
        self.nseries =243
        self.N_CLASSES = 6
    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        return self.data[ix], self.labels[ix]

    def load_dataset(self):
        data = pd.read_csv('data/time_series_data.csv',  header=None)
        labels = pd.read_csv('data/time_series_labels.csv',usecols=[0], header=None)

        self.nseries = 243
        print(self.nseries)
        # Create mapping of unique labels to integers
        label_map = {label: idx for idx, label in enumerate(labels[0].unique())}
        # Convert string labels to integers using the mapping
        labels = labels.replace(label_map)
        
        # Calculate number of complete groups of 80
        
        n_samples = len(data) // 80

        data_grouped = data.values.reshape(n_samples, 80, 24)
        labels_grouped = labels.values[::80]
        data = torch.tensor(np.asarray(data_grouped).astype(np.float32),
                            dtype=torch.float)
        print(data.shape)
        labels = torch.tensor(np.array(labels_grouped).astype(np.int32), dtype=torch.long)

        return data, labels.squeeze()
    