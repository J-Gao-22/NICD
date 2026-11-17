import numpy as np
from torch.utils.data import Dataset


class seq_dataset_eval(Dataset):
    def __init__(self, data, max_len=50):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_i = self.data[index]
        uid, iid, his, labels = data_i[0], data_i[1], data_i[2], data_i[3]
        if len(his) < self.max_len:
            his_ = np.zeros(self.max_len)
            if len(his):
                his_[-len(his):] = np.array(his)
            his = his_
        elif len(his) > self.max_len:
            his = np.array(his)[-self.max_len:]
        else:
            his = np.array(his)
        return uid, iid, his, labels
