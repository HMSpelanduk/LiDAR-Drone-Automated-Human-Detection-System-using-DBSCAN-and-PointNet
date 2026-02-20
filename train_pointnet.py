import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from prepare_dataset import MannequinPointCloudDataset
from tinypointnet2_model import TinyPointNetClassifier

# ==============================
# 設定
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TinyPointNetClassifier(num_classes=2)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# ==============================
# データセットとデータローダー
# ==============================

train_dataset = MannequinPointCloudDataset(root_dir="data_npy", num_points=2048)

# 元のバッチサイズは16
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# ==============================
# 学習ループ
# ==============================

for epoch in range(40):
    model.train()
    total_loss = 0

    for i, (points, labels) in enumerate(train_loader):
        points, labels = points.to(device), labels.to(device)

        # 入力形状を [B, 3, 2048] に変換
        points = points.permute(0, 2, 1)

        optimizer.zero_grad()
        outputs = model(points)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

# ==============================
# モデル保存
# ==============================

torch.save(model.state_dict(), "pointnet_mannequin_classifier.pth")
print("学習完了。モデルを保存しました。")