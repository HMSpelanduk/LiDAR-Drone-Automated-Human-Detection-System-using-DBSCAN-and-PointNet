import os
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# ----------- Settings -----------
PLY_FILE = "data/test data/2nddrone.ply"  # full scene
OUTPUT_DIR = "clusters_data"               # where cluster_*.ply will be saved

# Preprocessing
VOXEL_SIZE = 0.02 #for drone (0.02~0.04)/for realsense (0.0008~0.01)
PLANE_DIST_THRESH = 0.015 #for drone (0.015~0.02)/for realsense (0.004~0.006)
MAX_PLANES_TO_REMOVE = 1

# Outlier removal
USE_OUTLIER_REMOVAL = True
OUTLIER_NB_NEIGHBORS = 20
OUTLIER_STD_RATIO = 2.0

# DBSCAN
DBSCAN_EPS = 0.031 #for drone (0.03~0.2)/for realsense (0.023~0.035) // if mannequin and background mix, decrease value
DBSCAN_MIN_SAMPLES = 10 #for drone (10~20)/for realsense (40~60)
MIN_POINTS_IN_CLUSTER = 500 #original value 150

# Cluster merging
ENABLE_CLUSTER_MERGE = False
MERGE_DIST = 0.40 #for drone (0.20~0.40)/for realsense (0.08~0.12) // if mannequin breaks into many parts, increase

# Visualization
SHOW_GROUND = True
GROUND_COLOR = (0.6, 0.6, 0.6)
# --------------------------------


def segment_scene_with_dbscan(points,
                              eps=DBSCAN_EPS,
                              min_samples=DBSCAN_MIN_SAMPLES,
                              min_points=MIN_POINTS_IN_CLUSTER):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    unique_labels = set(labels)

    clusters = {}
    for label in unique_labels:
        if label == -1:
            continue
        cluster_pts = points[labels == label]
        if cluster_pts.shape[0] < min_points:
            continue
        clusters[label] = cluster_pts
    return clusters


def merge_close_clusters(clusters_dict, merge_dist=MERGE_DIST):
    labels = list(clusters_dict.keys())
    used = set()
    merged_clusters = []
    centroids = {lbl: clusters_dict[lbl].mean(axis=0) for lbl in labels}

    for i, lbl_i in enumerate(labels):
        if lbl_i in used:
            continue

        current_pts = clusters_dict[lbl_i]
        used.add(lbl_i)

        for lbl_j in labels[i + 1:]:
            if lbl_j in used:
                continue

            dist = np.linalg.norm(centroids[lbl_i] - centroids[lbl_j])
            if dist < merge_dist:
                current_pts = np.vstack((current_pts, clusters_dict[lbl_j]))
                used.add(lbl_j)

        merged_clusters.append(current_pts)

    return merged_clusters


def extract_dominant_planes(pcd,
                            dist_thresh=PLANE_DIST_THRESH,
                            max_planes=MAX_PLANES_TO_REMOVE):
    """
    Extract dominant planes (as a single combined point cloud) and return:
      - plane_pcd: all plane points (ground/wall)
      - non_plane_pcd: remaining points to cluster
    """
    current = pcd
    plane_parts = []

    for k in range(max_planes):
        if len(current.points) < 2000:
            break

        plane_model, inliers = current.segment_plane(
            distance_threshold=dist_thresh,
            ransac_n=3,
            num_iterations=3000
        )

        if len(inliers) < 0.25 * len(current.points):
            break

        plane_k = current.select_by_index(inliers)
        plane_parts.append(plane_k)

        current = current.select_by_index(inliers, invert=True)
        print(f"[INFO] Extracted plane {k+1}: plane_pts={len(plane_k.points)}, remaining={len(current.points)}")

    if len(plane_parts) > 0:
        plane_pcd = plane_parts[0]
        for p in plane_parts[1:]:
            plane_pcd += p
    else:
        plane_pcd = o3d.geometry.PointCloud()

    non_plane_pcd = current
    return plane_pcd, non_plane_pcd


def main():
    pcd = o3d.io.read_point_cloud(PLY_FILE)
    if len(pcd.points) == 0:
        print("[ERROR] Point cloud is empty.")
        return
    print(f"[INFO] Loaded {len(pcd.points)} points from {PLY_FILE}")

    # Downsample
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    print(f"[INFO] After voxel downsample: {len(pcd.points)} points")

    # Outlier removal
    if USE_OUTLIER_REMOVAL:
        pcd, ind = pcd.remove_statistical_outlier(
            nb_neighbors=OUTLIER_NB_NEIGHBORS,
            std_ratio=OUTLIER_STD_RATIO
        )
        print(f"[INFO] After outlier removal: {len(pcd.points)} points")

    # Extract planes but KEEP them for visualization
    plane_pcd, obj_pcd = extract_dominant_planes(pcd)

    obj_points = np.asarray(obj_pcd.points)
    if obj_points.shape[0] == 0:
        print("[ERROR] No object points left after plane extraction.")
        return

    print(f"[INFO] Object points for DBSCAN: {obj_points.shape[0]}")
    print(f"[DEBUG] Scale: min={obj_points.min(axis=0)}, max={obj_points.max(axis=0)}")

    # DBSCAN on non-plane points
    clusters_dict = segment_scene_with_dbscan(obj_points)
    print(f"[INFO] DBSCAN found {len(clusters_dict)} valid clusters")

    if ENABLE_CLUSTER_MERGE and len(clusters_dict) > 0:
        clusters_list = merge_close_clusters(clusters_dict, merge_dist=MERGE_DIST)
        print(f"[INFO] After merging: {len(clusters_list)} clusters")
    else:
        clusters_list = list(clusters_dict.values())

    # Prepare output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    colors = plt.get_cmap("tab10").colors
    vis_geoms = []

    # Add ground/plane for visualization
    if SHOW_GROUND and len(plane_pcd.points) > 0:
        plane_vis = o3d.geometry.PointCloud(plane_pcd)
        plane_vis.paint_uniform_color(GROUND_COLOR)
        vis_geoms.append(plane_vis)

    # Save clusters + add to visualization
    for i, cluster_pts in enumerate(clusters_list):
        print(f"[CLUSTER {i}] size: {cluster_pts.shape[0]} points")

        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster_pts)
        cluster_pcd.paint_uniform_color(colors[i % len(colors)])

        out_path = os.path.join(OUTPUT_DIR, f"cluster_{i}.ply")
        o3d.io.write_point_cloud(out_path, cluster_pcd)

        vis_geoms.append(cluster_pcd)

    print(f"[INFO] Saved {len(clusters_list)} clusters to '{OUTPUT_DIR}'")

    # Visualize: ground + clusters together
    if len(vis_geoms) > 0:
        o3d.visualization.draw_geometries(vis_geoms)


if __name__ == "__main__":
    main()
