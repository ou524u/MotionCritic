import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Assuming you have already defined MotionCritic, loss_func, and MLP

class motion_pair_dataset(Dataset):
    def __init__(self, motion_pair_list_name='motion_pair_list.pth'):
        self.data = torch.load(motion_pair_list_name)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.motion_pair_list)