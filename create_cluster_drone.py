import os
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# ----------- 設定 -----------
PLY_FILE = "data/test data/2nddrone.ply"  # フルシーン点群
OUTPUT_DIR = "clusters_data"              # cluster_*.ply の保存先

# 前処理
VOXEL_SIZE = 0.02  # drone用 (0.02~0.04) / realsense用 (0.0008~0.01)
PLANE_DIST_THRESH = 0.015  # drone用 (0.015~0.02) / realsense用 (0.004~0.006)
MAX_PLANES_TO_REMOVE = 1

# 外れ値除去
USE_OUTLIER_REMOVAL = True
OUTLIER_NB_NEIGHBORS = 20
OUTLIER_STD_RATIO = 2.0

# DBSCAN設定
DBSCAN_EPS = 0.031  # drone用 (0.03~0.2) / realsense用 (0.023~0.035)
                    # マネキンと背景が混ざる場合は値を小さくする
DBSCAN_MIN_SAMPLES = 10  # drone用 (10~20) / realsense用 (40~60)
MIN_POINTS_IN_CLUSTER = 500  # 元の値は150

# クラスタ統合
ENABLE_CLUSTER_MERGE = False
MERGE_DIST = 0.40  # drone用 (0.20~0.40) / realsense用 (0.08~0.12)
                   # マネキンが細かく分割される場合は値を大きくする

# 可視化
SHOW_GROUND = True
GROUND_COLOR = (0.6, 0.6, 0.6)
# --------------------------------


def segment_scene_with_dbscan(points,
                              eps=DBSCAN_EPS,
                              min_samples=DBSCAN_MIN_SAMPLES,
                              min_points=MIN_POINTS_IN_CLUSTER):
    # DBSCANでクラスタリングし、一定点数以上のクラスタのみ保持
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
    # 重心距離が近いクラスタを統合する
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
    主要な平面（地面・壁など）を抽出し、以下を返す：
      - plane_pcd: 抽出された全ての平面点群
      - non_plane_pcd: クラスタリング対象となる残り点群
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
        print("[ERROR] 点群が空です。")
        return
    print(f"[INFO] {PLY_FILE} から {len(pcd.points)} 点を読み込みました")

    # ダウンサンプリング
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    print(f"[INFO] ボクセルダウンサンプリング後: {len(pcd.points)} 点")

    # 外れ値除去
    if USE_OUTLIER_REMOVAL:
        pcd, ind = pcd.remove_statistical_outlier(
            nb_neighbors=OUTLIER_NB_NEIGHBORS,
            std_ratio=OUTLIER_STD_RATIO
        )
        print(f"[INFO] 外れ値除去後: {len(pcd.points)} 点")

    # 平面抽出（可視化用に保持）
    plane_pcd, obj_pcd = extract_dominant_planes(pcd)

    obj_points = np.asarray(obj_pcd.points)
    if obj_points.shape[0] == 0:
        print("[ERROR] 平面抽出後に物体点群が存在しません。")
        return

    print(f"[INFO] DBSCAN対象点数: {obj_points.shape[0]}")
    print(f"[DEBUG] スケール範囲: min={obj_points.min(axis=0)}, max={obj_points.max(axis=0)}")

    # 平面以外の点群に対してDBSCANを実行
    clusters_dict = segment_scene_with_dbscan(obj_points)
    print(f"[INFO] DBSCANにより {len(clusters_dict)} 個の有効クラスタを検出")

    if ENABLE_CLUSTER_MERGE and len(clusters_dict) > 0:
        clusters_list = merge_close_clusters(clusters_dict, merge_dist=MERGE_DIST)
        print(f"[INFO] 統合後クラスタ数: {len(clusters_list)}")
    else:
        clusters_list = list(clusters_dict.values())

    # 出力フォルダ作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    colors = plt.get_cmap("tab10").colors
    vis_geoms = []

    # 地面（平面）を可視化に追加
    if SHOW_GROUND and len(plane_pcd.points) > 0:
        plane_vis = o3d.geometry.PointCloud(plane_pcd)
        plane_vis.paint_uniform_color(GROUND_COLOR)
        vis_geoms.append(plane_vis)

    # 各クラスタを保存＋可視化
    for i, cluster_pts in enumerate(clusters_list):
        print(f"[CLUSTER {i}] 点数: {cluster_pts.shape[0]}")

        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster_pts)
        cluster_pcd.paint_uniform_color(colors[i % len(colors)])

        out_path = os.path.join(OUTPUT_DIR, f"cluster_{i}.ply")
        o3d.io.write_point_cloud(out_path, cluster_pcd)

        vis_geoms.append(cluster_pcd)

    print(f"[INFO] {len(clusters_list)} 個のクラスタを '{OUTPUT_DIR}' に保存しました")

    # 地面＋クラスタを同時に可視化
    if len(vis_geoms) > 0:
        o3d.visualization.draw_geometries(vis_geoms)


if __name__ == "__main__":
    main()