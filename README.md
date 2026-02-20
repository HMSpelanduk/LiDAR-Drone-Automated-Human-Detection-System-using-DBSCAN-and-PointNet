# Automated Human (Mannequin) Detection System using LiDAR Data

This repo contains a simple LiDAR point-cloud pipeline for **mannequin (human proxy) detection** using:
- **DBSCAN clustering** (object segmentation)
- **TinyPointNet** (lightweight PointNet-style classifier)
- **Open3D visualization** (clusters + 3D bounding box)

---

## Requirements (install first)

```bash
pip install numpy
pip install open3d
pip install scikit-learn
pip install matplotlib
pip install torch torchvision torchaudio

## Project Structure

Skeleton Training/
│
├── .venv/                      # Python virtual environment (ignored by git)
│
├── data/                       # Raw input point cloud data (ignored by git)
│   ├── mannequin/              # Mannequin object PLY samples
│   ├── background/             # Background object PLY samples
│   └── test data/              # Full test scenes (note: folder name has space)
│
├── data_npy/                   # Converted dataset in .npy format (ignored by git)
│   ├── mannequin/              # Mannequin samples (Nx3 numpy arrays)
│   └── background/             # Background samples (Nx3 numpy arrays)
│
├── clusters_data/              # Output DBSCAN clusters from full scenes (ignored)
│   └── cluster_*.ply
│
├── scratch code/               # Old / discarded scripts (kept for reference)
│
├── .gitignore                  # Git ignore rules (data, models, IDE files)
│
├── tinypointnet2_model.py      # TinyPointNet neural network definition
├── prepare_dataset.py          # PLY → NPY conversion + dataset loader
├── train_pointnet.py           # Train TinyPointNet classifier
├── check_model.py              # Simple script to verify trained model behavior
├── create_cluster_drone.py     # DBSCAN clustering for full scene point clouds
├── test_better_pointnet.py     # Main detection + visualization pipeline
│
└── pointnet_mannequin_classifier.pth
                                # Trained model weights (ignored by git)
