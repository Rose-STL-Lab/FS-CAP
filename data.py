import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import random
from copy import deepcopy, copy
import numpy as np


class FSCAPDataset(Dataset):
    def __init__(self, x, y, targets, context_ranges):
        target_to_idxs = {}
        for i in range(len(x)):
            if targets[i] not in target_to_idxs:
                target_to_idxs[targets[i]] = []
            target_to_idxs[targets[i]].append(i)
        y[y < 2] = 2
        y[y > 10] = 10
        for seq in target_to_idxs:
            seq_data = y[target_to_idxs[seq]].flatten()
            top_idxs = []
            for start, end in context_ranges:
                top_idxs.append(torch.arange(0, len(seq_data))[(seq_data >= start) & (seq_data < end)])
            top_idxs = [[target_to_idxs[seq][idx] for idx in idxs] for idxs in top_idxs]
            target_to_idxs[seq] = top_idxs
        self.x = x
        self.y = y
        self.targets = targets
        self.target_to_idxs = target_to_idxs
        self.avail_idxs = {target: [[]] * len(context_ranges) for target in self.target_to_idxs}

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        target = self.targets[idx]
        context_idxs = []
        for i in range(len(self.avail_idxs[target])):
            if not self.avail_idxs[target][i]:
                self.avail_idxs[target][i] = copy(self.target_to_idxs[target][i])
                random.shuffle(self.avail_idxs[target][i])
            context_idxs.append(self.avail_idxs[target][i].pop())
        return self.x[context_idxs], self.y[context_idxs], self.x[idx], self.y[idx], target


def get_dataloaders(batch_size, context_ranges):
    x_train, x_test, y_train, y_test, train_seqs, test_seqs = pickle.load(open('bindingdb_data.pickle', 'rb'))

    valid_seqs = []
    for line in open('clusterRes_rep_seq.fasta'):
        if not line.startswith('>'):
            valid_seqs.append(line.strip())
    valid_idxs = []
    for i in range(len(x_train)):
        if train_seqs[i] in valid_seqs:
            valid_idxs.append(i)
    valid_idxs = np.array(valid_idxs)
    x_train = x_train[valid_idxs]
    y_train = y_train[valid_idxs]
    train_seqs = [seq for seq in train_seqs if seq in valid_seqs]

    valid_idxs = []
    for i in range(len(x_test)):
        if test_seqs[i] in valid_seqs:
            valid_idxs.append(i)
    valid_idxs = np.array(valid_idxs)
    x_test = x_test[valid_idxs]
    y_test = y_test[valid_idxs]
    test_seqs = [seq for seq in test_seqs if seq in valid_seqs]
        
    train_dataloader = DataLoader(FSCAPDataset(x_train, y_train, train_seqs, context_ranges), batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(FSCAPDataset(x_test, y_test, test_seqs, context_ranges), batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader


def get_for_eval(data_file='bindingdb_data.pickle'):
    x_train, x_test, y_train, y_test, train_seqs, test_seqs, token_to_idx = pickle.load(open(data_file, 'rb'))
    return token_to_idx, y_train.mean(), y_train.std(), max(x_train.max(), x_test.max()) + 1, x_train.shape[1]