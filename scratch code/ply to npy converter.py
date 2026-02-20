import os
import numpy as np
import open3d as o3d


def ply_to_npy(input_dir, output_dir, normalize=True):
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if fname.endswith('.ply'):
            ply_path = os.path.join(input_dir, fname)
            pcd = o3d.io.read_point_cloud(ply_path)
            points = np.asarray(pcd.points)  # Nx3

            if normalize:
                # Center and scale to unit sphere
                centroid = np.mean(points, axis=0)
                points -= centroid
                furthest_dist = np.max(np.linalg.norm(points, axis=1))
                points /= furthest_dist

            npy_path = os.path.join(output_dir, fname.replace('.ply', '.npy'))
            np.save(npy_path, points)
            print(f"Saved {npy_path}")


# Example usage
ply_to_npy(
    input_dir='../data/mannequin',         # for mannequin files
    output_dir='../data_npy/mannequin'
)

ply_to_npy(
    input_dir='../data/background',        # for background files
    output_dir='../data_npy/background'
)


