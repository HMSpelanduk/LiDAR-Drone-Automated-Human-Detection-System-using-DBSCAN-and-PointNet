import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from prepare_dataset import MannequinPointCloudDataset
#from pointnet2_model import PointNet2ClassificationMSG #guna bila dataset byk
from tinypointnet2_model import TinyPointNetClassifier


# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = PointNet2ClassificationMSG(num_classes=2).to(device) #guna bila dataset byk
model = TinyPointNetClassifier(num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Dataset and DataLoader
train_dataset = MannequinPointCloudDataset(root_dir="data_npy", num_points=2048)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True) #original batch size 16

# Training loop
for epoch in range(40):
    model.train()
    total_loss = 0
    for i, (points, labels) in enumerate(train_loader):
        points, labels = points.to(device), labels.to(device)
        points = points.permute(0, 2, 1)  # [B, 3, 2048]
        optimizer.zero_grad()
        outputs = model(points)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

# Save model
torch.save(model.state_dict(), "pointnet_mannequin_classifier.pth")
print("Training complete. Model saved.")
