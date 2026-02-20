import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyPointNetClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # shared MLP on each point: 3 -> 64 -> 128
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)

        # global feature -> MLP
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):          # x: [B, 3, N]
        x = F.relu(self.conv1(x))  # [B, 64, N]
        x = F.relu(self.conv2(x))  # [B, 128, N]

        x = torch.max(x, dim=2)[0] # global max pool → [B, 128]

        x = F.relu(self.fc1(x))    # [B, 128]
        x = self.dropout(x)
        x = self.fc2(x)            # [B, num_classes]

        return x
