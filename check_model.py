import os, numpy as np, torch
from tinypointnet2_model import TinyPointNetClassifier

# ==============================
# モデル読み込み設定
# ==============================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TinyPointNetClassifier(num_classes=2).to(DEVICE)
model.load_state_dict(torch.load("pointnet_mannequin_classifier.pth", map_location=DEVICE))
model.eval()


# ==============================
# .npyファイルを分類する関数
# ==============================

def classify_npy(path):
    pc = np.load(path)           # (N, 3) の点群データを読み込み

    # 2048点にサンプリング（不足している場合は重複抽出）
    if pc.shape[0] >= 2048:
        idxs = np.random.choice(pc.shape[0], 2048, replace=False)
    else:
        idxs = np.random.choice(pc.shape[0], 2048, replace=True)

    pc = pc[idxs]

    # モデル入力形式に変換
    x = torch.from_numpy(pc).unsqueeze(0).float().to(DEVICE)  # (1, 2048, 3)
    x = x.permute(0, 2, 1)  # (1, 3, 2048)

    # 推論（勾配計算なし）
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

    # 予測クラス取得
    pred = probs.argmax(dim=1).item()
    cls = "background" if pred == 0 else "mannequin"

    print(path, "→ 確率 =", probs.cpu().numpy(), "→ 予測結果:", cls)


# ==============================
# 背景データでテスト
# ==============================

for f in os.listdir("data_npy/background"):
    classify_npy(os.path.join("data_npy/background", f))


# ==============================
# マネキンデータでテスト
# ==============================

for f in os.listdir("data_npy/mannequin"):
    classify_npy(os.path.join("data_npy/mannequin", f))