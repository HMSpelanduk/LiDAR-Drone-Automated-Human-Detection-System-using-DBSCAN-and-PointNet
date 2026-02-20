import os, numpy as np, torch
from tinypointnet2_model import TinyPointNetClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyPointNetClassifier(num_classes=2).to(DEVICE)
model.load_state_dict(torch.load("pointnet_mannequin_classifier.pth", map_location=DEVICE))
model.eval()

def classify_npy(path):
    pc = np.load(path)           # (N, 3)
    if pc.shape[0] >= 2048:
        idxs = np.random.choice(pc.shape[0], 2048, replace=False)
    else:
        idxs = np.random.choice(pc.shape[0], 2048, replace=True)
    pc = pc[idxs]
    x = torch.from_numpy(pc).unsqueeze(0).float().to(DEVICE)  # (1, 2048, 3)
    x = x.permute(0, 2, 1)  # (1, 3, 2048)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

    pred = probs.argmax(dim=1).item()
    cls = "background" if pred == 0 else "mannequin"
    print(path, "→ probs =", probs.cpu().numpy(), "→ predicted:", cls)


# test on training backgrounds
for f in os.listdir("data_npy/background"):
    classify_npy(os.path.join("data_npy/background", f))

# test on training mannequins
for f in os.listdir("data_npy/mannequin"):
    classify_npy(os.path.join("data_npy/mannequin", f))
