import os
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset

# ==============================
# STEP 1: PLY → NPY 変換処理
# ==============================

def ply_to_npy(input_dir, output_dir, normalize=True):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[STEP 1] PLY → NPY 変換開始")
    print(f"  input_dir  = {input_dir}")
    print(f"  output_dir = {output_dir}")

    ply_files = [f for f in os.listdir(input_dir) if f.endswith(".ply")]
    if not ply_files:
        print(f"  [WARN] {input_dir} に .ply ファイルが見つかりません")
        return

    for fname in ply_files:
        ply_path = os.path.join(input_dir, fname)
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)  # (N, 3)

        if points.size == 0:
            print(f"  [SKIP] {fname}: 点群が空です")
            continue

        if normalize:
            # 重心を原点に移動し、単位球内に正規化
            centroid = np.mean(points, axis=0)
            points = points - centroid
            furthest_dist = np.max(np.linalg.norm(points, axis=1))
            if furthest_dist > 0:
                points = points / furthest_dist

        npy_name = fname.replace(".ply", ".npy")
        npy_path = os.path.join(output_dir, npy_name)
        np.save(npy_path, points)
        print(f"  [OK] {fname}  →  {npy_name}")

    print(f"[DONE] {input_dir} 内の {len(ply_files)} 個のファイル変換が完了しました")


# ==============================
# STEP 2: データセット定義
# ==============================

class MannequinPointCloudDataset(Dataset):
    def __init__(self, root_dir, num_points=2048):
        self.num_points = num_points
        self.files = []
        self.labels = []

        # ラベル定義: 0 → background（背景）, 1 → mannequin（マネキン）
        for label, category in enumerate(["background", "mannequin"]):
            category_dir = os.path.join(root_dir, category)
            if not os.path.isdir(category_dir):
                print(f"[WARN] フォルダが見つかりません: {category_dir}")
                continue

            for fname in os.listdir(category_dir):
                if fname.endswith(".npy"):
                    self.files.append(os.path.join(category_dir, fname))
                    self.labels.append(label)

        print(f"\n[STEP 2] データセット作成完了: {root_dir}")
        print(f"  総サンプル数: {len(self.files)}")

        # クラスごとのサンプル数確認
        bg_count = sum(1 for l in self.labels if l == 0)
        man_count = sum(1 for l in self.labels if l == 1)
        print(f"  背景サンプル数 (label 0): {bg_count}")
        print(f"  マネキンサンプル数 (label 1): {man_count}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pc = np.load(self.files[idx])  # (N, 3)
        if pc.shape[0] >= self.num_points:
            idxs = np.random.choice(pc.shape[0], self.num_points, replace=False)
        else:
            idxs = np.random.choice(pc.shape[0], self.num_points, replace=True)
        pc = pc[idxs]
        label = self.labels[idx]
        return torch.from_numpy(pc).float(), label


# ==============================
# メイン処理パイプライン
# ==============================

def main():
    # ---- 1) すべてのPLYをNPYに変換 ----
    # マネキンクラス
    ply_to_npy(
        input_dir="data/mannequin",
        output_dir="data_npy/mannequin",
        normalize=True,
    )

    # 背景クラス
    ply_to_npy(
        input_dir="data/background",
        output_dir="data_npy/background",
        normalize=True,
    )

    # ---- 2) データセット構築とサンプル確認 ----
    dataset_root = "data_npy"
    dataset = MannequinPointCloudDataset(dataset_root, num_points=2048)

    if len(dataset) == 0:
        print("\n[ERROR] データセットが空です（.npy ファイルが見つかりません）。")
        return

    # サンプルを1つ取得して確認
    pc, label = dataset[0]
    print("\n[CHECK] データセット内のサンプル例:")
    print(f"  点群テンソルの形状: {pc.shape}")   # (2048, 3) のはず
    print(f"  ラベル: {label}")

    # CUDA動作確認用（必要な場合のみ使用）
    # pc = pc.unsqueeze(0).permute(0, 2, 1)  # (1, 3, 2048)
    # print("  モデル入力用形状:", pc.shape)

    print("\n[ALL DONE] PLY→NPY変換およびデータセット作成が正常に完了しました。")


if __name__ == "__main__":
    main()