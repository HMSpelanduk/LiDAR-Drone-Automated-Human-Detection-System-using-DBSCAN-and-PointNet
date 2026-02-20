import os
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset

# ==============================
# STEP 1: PLY → NPY CONVERTER;
# ==============================

def ply_to_npy(input_dir, output_dir, normalize=True):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[STEP 1] Converting PLY → NPY")
    print(f"  input_dir  = {input_dir}")
    print(f"  output_dir = {output_dir}")

    ply_files = [f for f in os.listdir(input_dir) if f.endswith(".ply")]
    if not ply_files:
        print(f"  [WARN] No .ply files found in {input_dir}")
        return

    for fname in ply_files:
        ply_path = os.path.join(input_dir, fname)
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)  # (N, 3)

        if points.size == 0:
            print(f"  [SKIP] {fname}: empty point cloud")
            continue

        if normalize:
            # Center and scale to unit sphere
            centroid = np.mean(points, axis=0)
            points = points - centroid
            furthest_dist = np.max(np.linalg.norm(points, axis=1))
            if furthest_dist > 0:
                points = points / furthest_dist

        npy_name = fname.replace(".ply", ".npy")
        npy_path = os.path.join(output_dir, npy_name)
        np.save(npy_path, points)
        print(f"  [OK] {fname}  →  {npy_name}")

    print(f"[DONE] Finished converting {len(ply_files)} file(s) in {input_dir}")


# ==============================
# STEP 2: DATASET DEFINITION
# ==============================

class MannequinPointCloudDataset(Dataset):
    def __init__(self, root_dir, num_points=2048):
        self.num_points = num_points
        self.files = []
        self.labels = []

        # label = 0 → background, 1 → mannequin
        for label, category in enumerate(["background", "mannequin"]):
            category_dir = os.path.join(root_dir, category)
            if not os.path.isdir(category_dir):
                print(f"[WARN] Missing folder: {category_dir}")
                continue

            for fname in os.listdir(category_dir):
                if fname.endswith(".npy"):
                    self.files.append(os.path.join(category_dir, fname))
                    self.labels.append(label)

        print(f"\n[STEP 2] Dataset created from: {root_dir}")
        print(f"  Total samples: {len(self.files)}")

        # quick per-class count
        bg_count = sum(1 for l in self.labels if l == 0)
        man_count = sum(1 for l in self.labels if l == 1)
        print(f"  Background samples (label 0): {bg_count}")
        print(f"  Mannequin  samples (label 1): {man_count}")

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
# MAIN PIPELINE
# ==============================

def main():
    # ---- 1) Convert all PLY to NPY ----
    # Mannequin class
    ply_to_npy(
        input_dir="data/mannequin",
        output_dir="data_npy/mannequin",
        normalize=True,
    )

    # Background class
    ply_to_npy(
        input_dir="data/background",
        output_dir="data_npy/background",
        normalize=True,
    )

    # ---- 2) Build dataset & show a sample ----
    dataset_root = "data_npy"
    dataset = MannequinPointCloudDataset(dataset_root, num_points=2048)

    if len(dataset) == 0:
        print("\n[ERROR] Dataset is empty (no .npy files found).")
        return

    # take one sample to inspect
    pc, label = dataset[0]
    print("\n[CHECK] Example sample from dataset:")
    print(f"  points tensor shape: {pc.shape}")   # should be (2048, 3)
    print(f"  label: {label}")

    # if you want to quickly make sure CUDA etc works:
    # pc = pc.unsqueeze(0).permute(0, 2, 1)  # (1, 3, 2048)
    # print("  ready for model input shape:", pc.shape)

    print("\n[ALL DONE] PLY→NPY + dataset creation finished successfully.")


if __name__ == "__main__":
    main()
