import os
import time
import numpy as np
import torch
import open3d as o3d
from sklearn.cluster import DBSCAN
from tinypointnet2_model import TinyPointNetClassifier
import matplotlib.pyplot as plt

# ============================================================
# 設定
# ============================================================

# このPythonファイルの場所を基準にパスを解決する
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(SCRIPT_DIR, "data")
TEST_DIR = os.path.join(DATA_DIR, "test data")
MODEL_PATH = os.path.join(SCRIPT_DIR, "pointnet_mannequin_classifier.pth")

# テスト用PLYファイル（必要に応じてファイル名のみ変更）
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
DBSCAN_EPS = 0.031
DBSCAN_MIN_SAMPLES = 10
MIN_POINTS_IN_CLUSTER = 500

ENABLE_CLUSTER_MERGE = False
MERGE_DIST = 0.08

# ------------------------------
# ステージ2 DBSCAN（再分割による精密化）
# ------------------------------
ENABLE_STAGE2 = False

# eps2（cmオーダー、単位はm）
STAGE2_EPS_MIN = 0.02
STAGE2_EPS_MAX = 0.10
STAGE2_HEIGHT_FACTOR = 0.03  # 例：H=1.7 → 約0.051m

STAGE2_MIN_SAMPLES = 8
STAGE2_MIN_POINTS = 80

STAGE2_SHRINK_STEPS = [1.00, 0.85, 0.70, 0.55, 0.45]
STAGE2_TARGET_MIN_CLUSTERS = 2
STAGE2_TARGET_MAX_CLUSTERS = 12

# ブリッジ除去（高速・時間制限あり）
ENABLE_BRIDGE_REMOVAL = True
BRIDGE_MIN_NEIGHBORS = 8
BRIDGE_MAX_SECONDS = 0.8
BRIDGE_DOWNSAMPLE_FACTOR = 0.5

# ステージ2で分割されすぎた場合の局所マージ（任意）
ENABLE_STAGE2_LOCAL_MERGE = True
LOCAL_MERGE_DIST = 0.035

# ------------------------------
# 可視化
# ------------------------------
SHOW_GROUND = True
GROUND_COLOR = (0.6, 0.6, 0.6)
NOISE_GRAY = (0.5, 0.5, 0.5)
MANNEQUIN_RED = (1.0, 0.0, 0.0)   # 赤

# ============================================================
# モデル
# ============================================================
def load_model(model_path: str):
    # モデルファイルの存在確認
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Model not found: {model_path}")

    m = TinyPointNetClassifier(num_classes=2).to(DEVICE)
    m.load_state_dict(torch.load(model_path, map_location=DEVICE))
    m.eval()
    return m

model = load_model(MODEL_PATH)

# ============================================================
# 補助関数
# ============================================================
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def ensure_finite_points(pts: np.ndarray, name: str = "points"):
    # 空配列、NaN/Infをチェック
    if pts is None or len(pts) == 0:
        raise ValueError(f"[ERROR] {name} is empty.")
    if not np.isfinite(pts).all():
        raise ValueError(f"[ERROR] {name} contains NaN/Inf.")

def aabb_dims(pts: np.ndarray) -> np.ndarray:
    # AABBの各辺長（dx, dy, dz）を返す
    ensure_finite_points(pts, "aabb_dims pts")
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    return mx - mn

def segment_scene_with_dbscan(points: np.ndarray, eps: float, min_samples: int, min_points: int):
    # DBSCANでクラスタリングし、一定点数以上のクラスタのみ返す
    ensure_finite_points(points, "DBSCAN input points")
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(points).labels_

    clusters = {}
    for lab in set(labels):
        if lab == -1:
            continue
        pts = points[labels == lab]
        if len(pts) >= min_points:
            clusters[lab] = pts

    noise = points[labels == -1] if np.any(labels == -1) else np.empty((0, 3), dtype=np.float64)
    return clusters, noise

def merge_close_clusters_list(clusters_list, merge_dist: float):
    # 重心距離が近いクラスタ同士を統合する（任意）
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
            if np.linalg.norm(centroids[i] - centroids[j]) < merge_dist:
                current = np.vstack((current, clusters_list[j]))
                used[j] = True

        merged.append(current)
    return merged

def extract_dominant_planes(pcd, dist_thresh=PLANE_DIST_THRESH, max_planes=MAX_PLANES_TO_REMOVE):
    # RANSACによる平面抽出（地面など）を行い、平面と残り点群を返す
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

    if plane_parts:
        plane = plane_parts[0]
        for p in plane_parts[1:]:
            plane += p
    else:
        plane = o3d.geometry.PointCloud()

    return plane, current

def normalize_and_sample(cluster_pts: np.ndarray, num_points=NUM_POINTS) -> np.ndarray:
    # 重心原点化 + スケール正規化 + 固定点数サンプリング
    ensure_finite_points(cluster_pts, "cluster_pts for normalize_and_sample")

    centroid = np.mean(cluster_pts, axis=0)
    pts = cluster_pts - centroid

    furthest = np.max(np.linalg.norm(pts, axis=1))
    if not np.isfinite(furthest) or furthest <= 0:
        pts = pts * 0.0
    else:
        pts = pts / furthest

    if len(pts) >= num_points:
        idx = np.random.choice(len(pts), num_points, replace=False)
    else:
        idx = np.random.choice(len(pts), num_points, replace=True)

    return pts[idx]

def classify_cluster(cluster_pts: np.ndarray):
    # PointNet系モデルで（背景/マネキン）確率を推定
    pc = normalize_and_sample(cluster_pts, NUM_POINTS)
    x = torch.from_numpy(pc).unsqueeze(0).float().to(DEVICE)  # [1,N,3]
    x = x.permute(0, 2, 1)  # [1,3,N]
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    return float(probs[0]), float(probs[1])  # (background, mannequin)

def break_bridges_fast(points: np.ndarray, radius: float, min_neighbors: int,
                      max_seconds: float = BRIDGE_MAX_SECONDS,
                      downsample_factor: float = BRIDGE_DOWNSAMPLE_FACTOR):
    """
    高速かつ安全なブリッジ除去：
      - 候補点群をダウンサンプルして計算量を削減
      - 半径近傍探索を行う（時間制限あり）
      - 除去が過剰、または時間超過の場合は元の点群を返す
    """
    if points is None or len(points) < 400:
        return points

    ensure_finite_points(points, "break_bridges points")

    ds_voxel = max(VOXEL_SIZE, radius * downsample_factor)
    pcd0 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).voxel_down_sample(ds_voxel)
    pts = np.asarray(pcd0.points)

    if pts is None or len(pts) < 200:
        return points

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    keep = np.zeros(len(pts), dtype=bool)
    start = time.time()

    for i, p in enumerate(pts):
        if (time.time() - start) > max_seconds:
            return points
        _, idx, _ = kdtree.search_radius_vector_3d(p, radius)
        if (len(idx) - 1) >= min_neighbors:
            keep[i] = True

    filtered = pts[keep]

    # 除去が過剰な場合は元点群を返す
    if filtered is None or len(filtered) < 0.60 * len(pts):
        return points

    return filtered

def stage2_refine_candidate(candidate_pts: np.ndarray):
    """
    ステージ2 DBSCAN（候補クラスタ内部の再分割）：
      - 候補クラスタのサイズHからeps2を算出
      - eps2を段階的に縮小して分割を試行
      - 必要に応じて高速ブリッジ除去を適用
    戻り値: (サブクラスタ一覧, 使用したeps2, H)
    """
    ensure_finite_points(candidate_pts, "stage2 candidate_pts")
    dims = aabb_dims(candidate_pts)
    H = float(dims.max())

    base_eps2 = clamp(STAGE2_HEIGHT_FACTOR * H, STAGE2_EPS_MIN, STAGE2_EPS_MAX)

    for s in STAGE2_SHRINK_STEPS:
        eps2 = clamp(base_eps2 * s, STAGE2_EPS_MIN, STAGE2_EPS_MAX)

        pts2 = candidate_pts
        if ENABLE_BRIDGE_REMOVAL:
            rad = max(VOXEL_SIZE, eps2)
            pts2 = break_bridges_fast(
                candidate_pts,
                radius=rad,
                min_neighbors=BRIDGE_MIN_NEIGHBORS,
                max_seconds=BRIDGE_MAX_SECONDS,
                downsample_factor=BRIDGE_DOWNSAMPLE_FACTOR
            )
            if pts2 is None or len(pts2) < STAGE2_MIN_POINTS:
                pts2 = candidate_pts

        sub_dict, _ = segment_scene_with_dbscan(
            pts2,
            eps=eps2,
            min_samples=STAGE2_MIN_SAMPLES,
            min_points=STAGE2_MIN_POINTS
        )
        subclusters = list(sub_dict.values())

        print(f"[DEBUG]   Stage2 try: base_eps2={base_eps2:.4f}, eps2={eps2:.4f}, "
              f"rad={max(VOXEL_SIZE, eps2):.4f}, pts2={len(pts2)}, subclusters={len(subclusters)}")

        # サブクラスタ数が妥当なら採用
        if STAGE2_TARGET_MIN_CLUSTERS <= len(subclusters) <= STAGE2_TARGET_MAX_CLUSTERS:
            return subclusters, eps2, H

    # 分割が得られない場合は元クラスタをそのまま返す
    return [candidate_pts], base_eps2, H

# ============================================================
# メイン処理
# ============================================================
# PLYファイルの存在確認
if not os.path.exists(PLY_FILE):
    raise FileNotFoundError(f"[ERROR] PLY file not found: {PLY_FILE}")

pcd_raw = o3d.io.read_point_cloud(PLY_FILE)
if len(pcd_raw.points) == 0:
    raise RuntimeError("[ERROR] Point cloud is empty.")
print(f"[INFO] Loaded {len(pcd_raw.points)} points from {PLY_FILE}")

# ボクセルダウンサンプリング
pcd = pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
print(f"[INFO] After voxel downsample: {len(pcd.points)} points")

# 外れ値除去（統計的手法）
if USE_OUTLIER_REMOVAL:
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=OUTLIER_NB_NEIGHBORS, std_ratio=OUTLIER_STD_RATIO)
    print(f"[INFO] After outlier removal: {len(pcd.points)} points")

# 平面抽出（地面など）と、物体点群への分離
plane_pcd, obj_pcd = extract_dominant_planes(pcd)
obj_points = np.asarray(obj_pcd.points)
print(f"[INFO] Object points for DBSCAN: {len(obj_points)}")
if len(obj_points) == 0:
    raise RuntimeError("[ERROR] No object points left after plane extraction.")

# ------------------------------
# ステージ1 DBSCAN（粗分割）
# ------------------------------
clusters_dict, noise_points = segment_scene_with_dbscan(
    obj_points,
    eps=DBSCAN_EPS,
    min_samples=DBSCAN_MIN_SAMPLES,
    min_points=MIN_POINTS_IN_CLUSTER
)
print(f"[INFO] Stage1 DBSCAN found {len(clusters_dict)} valid clusters")
print(f"[INFO] Stage1 DBSCAN noise points: {len(noise_points)}")

if ENABLE_CLUSTER_MERGE and clusters_dict:
    clusters_list = merge_close_clusters_list(list(clusters_dict.values()), merge_dist=MERGE_DIST)
    print(f"[INFO] After merging: {len(clusters_list)} clusters")
else:
    clusters_list = list(clusters_dict.values())

# ------------------------------
# 検出処理：マネキン候補を全て収集
# ------------------------------
detections = []

for i, cluster_pts in enumerate(clusters_list):
    p_back, p_man = classify_cluster(cluster_pts)
    pred = 1 if p_man >= p_back else 0
    print(f"[DEBUG] Stage1 Cluster {i}: pred={pred}, P(back)={p_back:.2f}, P(man)={p_man:.2f}, size={len(cluster_pts)}")

    if pred != 1 or p_man < CONFIDENCE_THRESHOLD:
        continue

    # ステージ2：候補クラスタ内部を再分割して精密化
    if ENABLE_STAGE2:
        subclusters, eps2, H = stage2_refine_candidate(cluster_pts)
        print(f"[DEBUG]   Stage2 selected: H={H:.3f} m, eps2_used={eps2:.4f}, subclusters={len(subclusters)}")
    else:
        subclusters, eps2, H = [cluster_pts], None, None

    # サブクラスタの中からマネキン確率が最大のものを選択（なければ候補全体）
    best = {"idx": None, "p_man": -1.0, "pts": None}
    for sj, sub_pts in enumerate(subclusters):
        if sub_pts is None or len(sub_pts) < 10:
            continue
        pb2, pm2 = classify_cluster(sub_pts)
        pred2 = 1 if pm2 >= pb2 else 0
        print(f"[DEBUG]     Sub {sj}: pred={pred2}, P(man)={pm2:.2f}, size={len(sub_pts)}")
        if pred2 == 1 and pm2 > best["p_man"]:
            best = {"idx": sj, "p_man": pm2, "pts": sub_pts}

    chosen_pts = best["pts"] if best["pts"] is not None else cluster_pts
    chosen_p = best["p_man"] if best["pts"] is not None else p_man
    chosen_sj = best["idx"]

    # 局所マージ（分割されたマネキンの近傍パーツを統合）
    if ENABLE_STAGE2 and ENABLE_STAGE2_LOCAL_MERGE and chosen_sj is not None and len(subclusters) > 1:
        chosen_centroid = chosen_pts.mean(axis=0)
        merged = [chosen_pts]
        for sj, sub_pts in enumerate(subclusters):
            if sj == chosen_sj:
                continue
            if np.linalg.norm(sub_pts.mean(axis=0) - chosen_centroid) < LOCAL_MERGE_DIST:
                merged.append(sub_pts)
        if len(merged) > 1:
            chosen_pts = np.vstack(merged)
            print(f"[DEBUG]   Local-merge: merged {len(merged)} subclusters -> size={len(chosen_pts)}")

    detections.append({
        "stage1_idx": i,
        "stage2_idx": chosen_sj,
        "p_man": chosen_p,
        "pts": chosen_pts,
        "reason": "NN+Stage2" if ENABLE_STAGE2 else "NN",
        "eps2": eps2,
        "H": H
    })

print(f"[INFO] Total mannequin detections above threshold: {len(detections)}")

# ============================================================
# 可視化
# ============================================================
geoms = []

# 地面（平面）
if SHOW_GROUND and len(plane_pcd.points) > 0:
    plane_vis = o3d.geometry.PointCloud(plane_pcd)
    plane_vis.paint_uniform_color(GROUND_COLOR)
    geoms.append(plane_vis)

# ノイズ点
if len(noise_points) > 0:
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
    cl_pcd.paint_uniform_color(cmap(i % 20)[:3])
    cluster_pcds.append(cl_pcd)
geoms += cluster_pcds

if len(detections) == 0:
    print("[RESULT] No mannequin detected above threshold.")
else:
    # 信頼度順にソートして全検出結果を表示
    detections.sort(key=lambda d: d["p_man"], reverse=True)

    for k, det in enumerate(detections):
        s1 = det["stage1_idx"]
        s2 = det["stage2_idx"]

        if ENABLE_STAGE2:
            print(f"[RESULT] #{k+1} Mannequin = Stage1[{s1}] Stage2[{s2}]  "
                  f"P(man)={det['p_man']:.2f} via {det['reason']} "
                  f"(H={det['H']:.3f}m, eps2={det['eps2']:.4f})")
        else:
            print(f"[RESULT] #{k+1} Mannequin = Stage1[{s1}]  P(man)={det['p_man']:.2f} via {det['reason']}")

        # 該当クラスタを赤で強調
        if 0 <= s1 < len(cluster_pcds):
            cluster_pcds[s1].paint_uniform_color(MANNEQUIN_RED)

        mannequin_pcd = o3d.geometry.PointCloud()
        mannequin_pcd.points = o3d.utility.Vector3dVector(det["pts"])
        mannequin_pcd.paint_uniform_color(MANNEQUIN_RED)
        geoms.append(mannequin_pcd)

        bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(det["pts"])
        )
        bbox.color = (0, 1, 0)
        geoms.append(bbox)

        dims = aabb_dims(det["pts"])
        print(f"[INFO] #{k+1} AABB dims: dx={dims[0]:.3f}, dy={dims[1]:.3f}, dz={dims[2]:.3f} (max={dims.max():.3f} m)")

o3d.visualization.draw_geometries(geoms)
