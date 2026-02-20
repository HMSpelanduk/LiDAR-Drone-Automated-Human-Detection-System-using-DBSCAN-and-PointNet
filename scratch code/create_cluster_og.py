import os
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# ----------- Settings -----------
PLY_FILE = "data/test_2.ply"   # full scene
OUTPUT_DIR = "../clusters_data"  # where cluster_*.ply will be saved
DBSCAN_EPS = 0.03         # tune based on your scene scale og:0.018
# if mannequin split into pieces, up value. background merges with mannequin, down value
DBSCAN_MIN_SAMPLES = 40 #og:30
MIN_POINTS_IN_CLUSTER = 150 #og:100
# --------------------------------


def segment_scene_with_dbscan(points,
                              eps=DBSCAN_EPS,
                              min_samples=DBSCAN_MIN_SAMPLES,
                              min_points=MIN_POINTS_IN_CLUSTER):
    """
    Run DBSCAN on a point cloud and return a dict: label -> (N_i, 3) array.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    unique_labels = set(labels)

    clusters = {}
    for label in unique_labels:
        if label == -1:
            continue  # noise
        cluster_pts = points[labels == label]
        if cluster_pts.shape[0] < min_points:
            continue
        clusters[label] = cluster_pts

    return clusters


def main():
    # Load scene
    pcd = o3d.io.read_point_cloud(PLY_FILE)
    points = np.asarray(pcd.points)

    if points.shape[0] == 0:
        print("[ERROR] Point cloud is empty.")
        return

    print(f"[INFO] Loaded {points.shape[0]} points from {PLY_FILE}")
    print(f"[DEBUG] Scale: min={points.min(axis=0)}, max={points.max(axis=0)}")

    # Run DBSCAN
    clusters = segment_scene_with_dbscan(points)
    print(f"[INFO] DBSCAN found {len(clusters)} valid clusters")

    # Prepare output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    colors = plt.get_cmap("tab10").colors

    colored_clusters = []

    # Save each cluster as its own .ply
    for i, (label, cluster_pts) in enumerate(clusters.items()):
        print(f"[CLUSTER {label}] size: {cluster_pts.shape[0]} points")

        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster_pts)
        cluster_pcd.paint_uniform_color(colors[label % len(colors)])

        out_path = os.path.join(OUTPUT_DIR, f"cluster_{label}.ply")
        o3d.io.write_point_cloud(out_path, cluster_pcd)
        colored_clusters.append(cluster_pcd)

    print(f"[INFO] Saved {len(colored_clusters)} clusters to '{OUTPUT_DIR}'")
    # Optional visualization
    if colored_clusters:
        o3d.visualization.draw_geometries(colored_clusters)


if __name__ == "__main__":
    main()
