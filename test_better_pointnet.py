import os
import numpy as np
import torch
import open3d as o3d
from sklearn.cluster import DBSCAN
from tinypointnet2_model import TinyPointNetClassifier
import matplotlib.pyplot as plt

# ==============================
# 設定
# ==============================

# このPythonファイルの場所を基準にパスを解決する
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(SCRIPT_DIR, "data")
TEST_DIR = os.path.join(DATA_DIR, "test data")  # フォルダ名にスペースあり → 正確に一致させる必要あり
MODEL_PATH = os.path.join(SCRIPT_DIR, "pointnet_mannequin_classifier.pth")

# オプションA：ファイル名で1つ選択（ファイル名だけ変更すればOK）
TEST_PLY_NAME = "2nddrone.ply"
PLY_FILE = os.path.join(TEST_DIR, TEST_PLY_NAME)

NUM_POINTS = 2048
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.8  # マネキン判定の信頼度しきい値

# --- 前処理 ---
VOXEL_SIZE = 0.02
PLANE_DIST_THRESH = 0.015
MAX_PLANES_TO_REMOVE = 1

USE_OUTLIER_REMOVAL = True
OUTLIER_NB_NEIGHBORS = 20
OUTLIER_STD_RATIO = 2.0

# ------------------------------
# ステージ1 DBSCAN（粗いクラスタリング）
# ------------------------------
DBSCAN_EPS = 0.031  # ドローン用 (0.04~0.2) / RealSense用 (0.023~0.035)
                    # マネキンと背景が混ざる場合は値を小さくする
DBSCAN_MIN_SAMPLES = 10
MIN_POINTS_IN_CLUSTER = 500

ENABLE_CLUSTER_MERGE = False
MERGE_DIST = 0.08

# ------------------------------
# ステージ2 DBSCAN（マネキン候補内部の再分割）
# ミニマネキン（約0.5m）の場合：eps2は0.6cm～1.6cm程度が目安
# ------------------------------
ENABLE_STAGE2 = False  # 動作がまだ安定していないため、通常は無効で使用する

STAGE2_EPS_MIN = 0.006        # 0.6 cm
STAGE2_EPS_MAX = 0.016        # 1.6 cm
STAGE2_HEIGHT_FACTOR = 0.025  # 基本eps2 = clamp(係数 × 高さ, 最小値, 最大値)

STAGE2_MIN_SAMPLES = 10
STAGE2_MIN_POINTS = 60        # 小さめのサブクラスタも許容

# ステージ2：eps2を段階的に縮小して分割を試みる
STAGE2_SHRINK_STEPS = [1.00, 0.85, 0.70, 0.55, 0.45]  # 分割されるまでeps2を徐々に小さくする
STAGE2_TARGET_MIN_CLUSTERS = 2
STAGE2_TARGET_MAX_CLUSTERS = 10

# マネキンと近傍物体が細い点群で繋がる場合、その「ブリッジ」を除去して分離しやすくする
ENABLE_BRIDGE_REMOVAL = True
BRIDGE_MIN_NEIGHBORS = 8
# 半径は関数内で max(VOXEL_SIZE, eps2) として決定する

# マネキンが複数パーツに分割された場合、近傍のサブクラスタを局所的に統合する
ENABLE_STAGE2_LOCAL_MERGE = True
LOCAL_MERGE_DIST = 0.035  # 3.5 cm（分割が細かすぎる場合に調整）

# ------------------------------
# 可視化
# ------------------------------
SHOW_GROUND = False
GROUND_COLOR = (0.6, 0.6, 0.6)
NOISE_GRAY = (0.5, 0.5, 0.5)
MANNEQUIN_BLUE = (0.1, 0.4, 1.0)

# ==============================
# モデル
# ==============================
model = TinyPointNetClassifier(num_classes=2).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ==============================
# 補助関数
# ==============================
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def aabb_dims(pts: np.ndarray) -> np.ndarray:
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    return mx - mn

def segment_scene_with_dbscan(points: np.ndarray, eps: float, min_samples: int, min_points: int):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    clusters_dict = {}

    for label in set(labels):
        if label == -1:
            continue
        cluster_pts = points[labels == label]
        if cluster_pts.shape[0] < min_points:
            continue
        clusters_dict[label] = cluster_pts

    noise_points = points[labels == -1] if np.any(labels == -1) else np.empty((0, 3), dtype=np.float64)
    return clusters_dict, noise_points

def merge_close_clusters_list(clusters_list, merge_dist: float):
    if len(clusters_list) <= 1:
        return clusters_list

    centroids = [c.mean(axis=0) for c in clusters_list]
    used = [False] * len(clusters_list)
    merged = []

    for i in range(len(clusters_list)):
        if used[i]:
            continue
        used[i] = True
        current = clusters_list[i]

        for j in range(i + 1, len(clusters_list)):
            if used[j]:
                continue
            d = np.linalg.norm(centroids[i] - centroids[j])
            if d < merge_dist:
                current = np.vstack((current, clusters_list[j]))
                used[j] = True

        merged.append(current)
    return merged

def extract_dominant_planes(pcd, dist_thresh=PLANE_DIST_THRESH, max_planes=MAX_PLANES_TO_REMOVE):
    current = pcd
    plane_parts = []

    for k in range(max_planes):
        if len(current.points) < 2000:
            break

        _, inliers = current.segment_plane(
            distance_threshold=dist_thresh,
            ransac_n=3,
            num_iterations=3000
        )

        if len(inliers) < 0.25 * len(current.points):
            break

        plane_k = current.select_by_index(inliers)
        plane_parts.append(plane_k)
        current = current.select_by_index(inliers, invert=True)
        print(f"[INFO] Extracted plane {k + 1}: plane_pts={len(plane_k.points)}, remaining={len(current.points)}")

    if len(plane_parts) > 0:
        plane_pcd = plane_parts[0]
        for p in plane_parts[1:]:
            plane_pcd += p
    else:
        plane_pcd = o3d.geometry.PointCloud()

    return plane_pcd, current

def normalize_and_sample(cluster_pts, num_points=NUM_POINTS):
    centroid = np.mean(cluster_pts, axis=0)
    pts = cluster_pts - centroid
    furthest_dist = np.max(np.linalg.norm(pts, axis=1))
    if furthest_dist > 0:
        pts = pts / furthest_dist

    if pts.shape[0] >= num_points:
        idxs = np.random.choice(pts.shape[0], num_points, replace=False)
    else:
        idxs = np.random.choice(pts.shape[0], num_points, replace=True)
    return pts[idxs]

def classify_cluster(cluster_pts):
    pc = normalize_and_sample(cluster_pts, NUM_POINTS)
    x = torch.from_numpy(pc).unsqueeze(0).float().to(DEVICE)  # [1,N,3]
    x = x.permute(0, 2, 1)  # [1,3,N]
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    return float(probs[0]), float(probs[1])

def break_bridges(points: np.ndarray, radius: float, min_neighbors: int):
    """
    2つの物体が疎な点群で繋がっている場合、その「ブリッジ」点を除去する。
    半径内の近傍点数（自身を除く）が min_neighbors 以上の点のみ保持する。
    """
    if len(points) < 200:
        return points

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    keep = np.zeros(len(points), dtype=bool)
    for i, p in enumerate(points):
        _, idx, _ = kdtree.search_radius_vector_3d(p, radius)
        # idx には自身も含まれる
        if (len(idx) - 1) >= min_neighbors:
            keep[i] = True

    filtered = points[keep]

    # 安全措置：除去が過剰な場合は元の点群を返す
    if len(filtered) < 0.60 * len(points):
        return points
    return filtered

def stage2_refine_candidate(candidate_pts: np.ndarray):
    """
    ステージ2 DBSCAN：
      - 候補クラスタのサイズH（AABB最大辺）から基本eps2を算出
      - eps2を段階的に縮小し、適切な数（2～10）のサブクラスタに分割できるか試行
      - 必要に応じて、クラスタリング前にブリッジ除去を実施
    戻り値: (サブクラスタ群, 使用したeps2, H)
    """
    dims = aabb_dims(candidate_pts)
    H = float(dims.max())  # サイズの指標（ミニマネキンで良好に機能）

    base_eps2 = clamp(STAGE2_HEIGHT_FACTOR * H, STAGE2_EPS_MIN, STAGE2_EPS_MAX)

    for s in STAGE2_SHRINK_STEPS:
        eps2 = clamp(base_eps2 * s, STAGE2_EPS_MIN, STAGE2_EPS_MAX)

        pts2 = candidate_pts
        if ENABLE_BRIDGE_REMOVAL:
            rad = max(VOXEL_SIZE, eps2)  # ボクセルサイズと整合のある半径にする
            pts2 = break_bridges(candidate_pts, radius=rad, min_neighbors=BRIDGE_MIN_NEIGHBORS)

        sub_dict, _ = segment_scene_with_dbscan(
            pts2,
            eps=eps2,
            min_samples=STAGE2_MIN_SAMPLES,
            min_points=STAGE2_MIN_POINTS
        )
        subclusters = list(sub_dict.values())

        print(f"[DEBUG]   Stage2 try: eps2={eps2:.4f}, pts2={len(pts2)}, subclusters={len(subclusters)}")

        # 分割数が妥当なら採用
        if STAGE2_TARGET_MIN_CLUSTERS <= len(subclusters) <= STAGE2_TARGET_MAX_CLUSTERS:
            return subclusters, eps2, H

    # 分割が得られない場合は、元クラスタをそのまま返す
    return [candidate_pts], base_eps2, H

# ==============================
# メイン処理
# ==============================
pcd_raw = o3d.io.read_point_cloud(PLY_FILE)
if len(pcd_raw.points) == 0:
    raise RuntimeError("[ERROR] Point cloud is empty.")
print(f"[INFO] Loaded {len(pcd_raw.points)} points from {PLY_FILE}")

pcd = pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
print(f"[INFO] After voxel downsample: {len(pcd.points)} points")

if USE_OUTLIER_REMOVAL:
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=OUTLIER_NB_NEIGHBORS, std_ratio=OUTLIER_STD_RATIO)
    print(f"[INFO] After outlier removal: {len(pcd.points)} points")

plane_pcd, obj_pcd = extract_dominant_planes(pcd)
obj_points = np.asarray(obj_pcd.points)
print(f"[INFO] Object points for DBSCAN: {obj_points.shape[0]}")
if obj_points.shape[0] == 0:
    raise RuntimeError("[ERROR] No object points left after plane extraction.")

# ==============================
# ステージ1 DBSCAN（粗分割）
# ==============================
clusters_dict, noise_points = segment_scene_with_dbscan(
    obj_points,
    eps=DBSCAN_EPS,
    min_samples=DBSCAN_MIN_SAMPLES,
    min_points=MIN_POINTS_IN_CLUSTER
)
print(f"[INFO] Stage1 DBSCAN found {len(clusters_dict)} valid clusters")
print(f"[INFO] Stage1 DBSCAN noise points: {noise_points.shape[0]}")

if ENABLE_CLUSTER_MERGE and len(clusters_dict) > 0:
    clusters_list = merge_close_clusters_list(list(clusters_dict.values()), merge_dist=MERGE_DIST)
    print(f"[INFO] After merging: {len(clusters_list)} clusters")
else:
    clusters_list = list(clusters_dict.values())

# ==============================
# 検出処理：
# ステージ1 → ステージ2 → 最良サブクラスタ選択 → バウンディングボックス生成
# ==============================
best = {
    "stage1_idx": None,
    "stage2_idx": None,
    "p_man": -1.0,
    "pts": None,
    "reason": "",
    "eps2": None,
    "H": None
}

for i, cluster_pts in enumerate(clusters_list):
    p_back, p_man = classify_cluster(cluster_pts)
    pred = 1 if p_man >= p_back else 0
    print(f"[DEBUG] Stage1 Cluster {i}: pred={pred}, P(back)={p_back:.2f}, P(man)={p_man:.2f}, size={len(cluster_pts)}")

    if pred != 1 or p_man < CONFIDENCE_THRESHOLD:
        continue

    if ENABLE_STAGE2:
        subclusters, eps2, H = stage2_refine_candidate(cluster_pts)
        print(f"[DEBUG]   Stage2 selected: H={H:.3f} m, eps2_used={eps2:.4f}, subclusters={len(subclusters)}")
    else:
        subclusters, eps2, H = [cluster_pts], None, None

    # マネキン確率が最大となるサブクラスタを選択
    local_best = {"stage2_idx": None, "p_man": -1.0, "pts": None}

    for sj, sub_pts in enumerate(subclusters):
        pb2, pm2 = classify_cluster(sub_pts)
        pred2 = 1 if pm2 >= pb2 else 0
        print(f"[DEBUG]     Sub {sj}: pred={pred2}, P(man)={pm2:.2f}, size={len(sub_pts)}")

        if pred2 == 1 and pm2 > local_best["p_man"]:
            local_best = {"stage2_idx": sj, "p_man": pm2, "pts": sub_pts}

    chosen_pts = local_best["pts"] if local_best["pts"] is not None else cluster_pts
    chosen_p = local_best["p_man"] if local_best["pts"] is not None else p_man
    chosen_sj = local_best["stage2_idx"]

    # 局所統合（マネキンが複数パーツに分割された場合の補助）
    if ENABLE_STAGE2 and ENABLE_STAGE2_LOCAL_MERGE and chosen_sj is not None and len(subclusters) > 1:
        merged = [chosen_pts]
        chosen_centroid = chosen_pts.mean(axis=0)

        for sj, sub_pts in enumerate(subclusters):
            if sj == chosen_sj:
                continue
            c = sub_pts.mean(axis=0)
            if np.linalg.norm(c - chosen_centroid) < LOCAL_MERGE_DIST:
                merged.append(sub_pts)

        if len(merged) > 1:
            chosen_pts = np.vstack(merged)
            print(f"[DEBUG]   Local-merge: merged {len(merged)} subclusters -> size={len(chosen_pts)}")

    if chosen_p > best["p_man"]:
        best = {
            "stage1_idx": i,
            "stage2_idx": chosen_sj,
            "p_man": chosen_p,
            "pts": chosen_pts,
            "reason": "NN+Stage2(auto-split)" if ENABLE_STAGE2 else "NN",
            "eps2": eps2,
            "H": H
        }

# ==============================
# 可視化
# ==============================
geoms = []

# 地面（平面）
if SHOW_GROUND and len(plane_pcd.points) > 0:
    plane_vis = o3d.geometry.PointCloud(plane_pcd)
    plane_vis.paint_uniform_color(GROUND_COLOR)
    geoms.append(plane_vis)

# ノイズ点
if noise_points.shape[0] > 0:
    noise_pcd = o3d.geometry.PointCloud()
    noise_pcd.points = o3d.utility.Vector3dVector(noise_points)
    noise_pcd.paint_uniform_color(NOISE_GRAY)
    geoms.append(noise_pcd)

# ステージ1の全クラスタを色分け表示
cmap = plt.get_cmap("tab20")
cluster_pcds = []
for i, cluster_pts in enumerate(clusters_list):
    cl_pcd = o3d.geometry.PointCloud()
    cl_pcd.points = o3d.utility.Vector3dVector(cluster_pts)
    color = cmap(i % 20)[:3]
    cl_pcd.paint_uniform_color(color)
    cluster_pcds.append(cl_pcd)
geoms += cluster_pcds

# 選択されたマネキン点群のみバウンディングボックスを表示
if best["pts"] is not None and best["stage1_idx"] is not None:
    s1 = best["stage1_idx"]
    s2 = best["stage2_idx"]

    if ENABLE_STAGE2:
        print(f"[RESULT] Mannequin = Stage1[{s1}] Stage2[{s2}]  P(man)={best['p_man']:.2f} via {best['reason']}  (H={best['H']:.3f}m, eps2={best['eps2']:.4f})")
    else:
        print(f"[RESULT] Mannequin = Stage1[{s1}]  P(man)={best['p_man']:.2f} via {best['reason']}")

    # マネキン候補クラスタを青色で強調
    if 0 <= s1 < len(cluster_pcds):
        cluster_pcds[s1].paint_uniform_color(MANNEQUIN_BLUE)

    mannequin_pcd = o3d.geometry.PointCloud()
    mannequin_pcd.points = o3d.utility.Vector3dVector(best["pts"])
    mannequin_pcd.paint_uniform_color(MANNEQUIN_BLUE)
    geoms.append(mannequin_pcd)

    bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(best["pts"])
    )
    bbox.color = (0, 1, 0)
    geoms.append(bbox)

    dims = aabb_dims(best["pts"])
    print(f"[INFO] Final mannequin AABB dims: dx={dims[0]:.3f}, dy={dims[1]:.3f}, dz={dims[2]:.3f} (max={dims.max():.3f} m)")
else:
    print("[RESULT] No mannequin detected above threshold.")

o3d.visualization.draw_geometries(geoms)
