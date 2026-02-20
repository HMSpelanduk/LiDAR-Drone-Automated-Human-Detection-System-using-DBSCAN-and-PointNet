import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MannequinPointCloudDataset(Dataset):
    def __init__(self, root_dir, num_points=2048):
        self.num_points = num_points
        self.files = []
        self.labels = []

        for label, category in enumerate(['background', 'mannequin']):
            category_dir = os.path.join(root_dir, category)
            for fname in os.listdir(category_dir):
                if fname.endswith('.npy'):
                    self.files.append(os.path.join(category_dir, fname))
                    self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pc = np.load(self.files[idx])
        if pc.shape[0] >= self.num_points:
            idxs = np.random.choice(pc.shape[0], self.num_points, replace=False)
        else:
            idxs = np.random.choice(pc.shape[0], self.num_points, replace=True)
        pc = pc[idxs]
        label = self.labels[idx]
        return torch.from_numpy(pc).float(), label
