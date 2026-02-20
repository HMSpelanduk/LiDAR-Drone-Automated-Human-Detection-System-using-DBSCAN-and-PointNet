import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, BatchNorm1d as BN

class PointNet2ClassificationMSG(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(3, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(128, num_classes)

    def forward(self, x):  # x: [B, 3, N]
        x = x.transpose(1, 2)  # [B, N, 3]
        x = torch.mean(x, dim=1)  # [B, 3] → global feature

        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.drop3(F.relu(self.bn3(self.fc3(x))))
        x = self.fc4(x)

        return x
