import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import random
from copy import copy


class FSCAPDataset(Dataset):
    def __init__(self, x, y, targets, context_ranges):
        target_to_idxs = {}
        for i in range(len(x)):
            if targets[i] not in target_to_idxs:
                target_to_idxs[targets[i]] = []
            target_to_idxs[targets[i]].append(i)
        y[y < -2.5] = -2.5
        y[y > 6.5] = 6.5
        for assay in target_to_idxs:
            assay_data = y[target_to_idxs[assay]].flatten()
            top_idxs = []
            for start, end in context_ranges:
                top_idxs.append(torch.arange(0, len(assay_data))[(assay_data >= start) & (assay_data < end)])
            top_idxs = [[target_to_idxs[assay][idx] for idx in idxs] for idxs in top_idxs]
            target_to_idxs[assay] = top_idxs
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


def get_dataloaders(batch_size, context_ranges, data_file):
    x_train, x_test, y_train, y_test, train_assays, test_assays = pickle.load(open(f'{data_file}_data.pickle', 'rb'))
    train_dataloader = DataLoader(FSCAPDataset(x_train, y_train, train_assays, context_ranges), batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(FSCAPDataset(x_test, y_test, test_assays, context_ranges), batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader
