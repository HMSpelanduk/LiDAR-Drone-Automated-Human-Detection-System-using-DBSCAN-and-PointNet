import os
import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import procrustes

from tinypointnet2_model import TinyPointNetClassifier

# ----------- Settings ----------- #
PLY_FILE = "../data/test data/test_3.ply"  # full scene to evaluate
MODEL_PATH = "../pointnet_mannequin_classifier.pth"  # trained PointNet model

SKELETON_FOLDER = "manual_skeletons"          # cluster_X_skeleton.npy live here
REFERENCE_TEMPLATES_FOLDER = "reference_templates"  # many mannequin skeleton templates (.npy)

NUM_POINTS = 2048
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Thresholds
P_MANNEQUIN_MIN_FOR_CHECK = 0.30   # min P(man) to even TRY skeleton check
CONFIDENCE_FOR_YELLOW_BOX = 0.60   # P(man) needed to draw classifier-only (yellow) bbox
SKELETON_MATCH_THRESHOLD = 0.05    # Procrustes disparity threshold

# DBSCAN settings
DBSCAN_EPS = 0.02
DBSCAN_MIN_SAMPLES = 30
MIN_POINTS_IN_CLUSTER = 100
# -------------------------------- #


def segment_scene_with_dbscan(points,
                              eps=DBSCAN_EPS,
                              min_samples=DBSCAN_MIN_SAMPLES,
                              min_points=MIN_POINTS_IN_CLUSTER):
    print("[DEBUG] Starting DBSCAN...")
    clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(points)
    print("[DEBUG] Finished DBSCAN.")
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


def is_skeleton_match_multi(candidate_skeleton_path,
                            templates_folder=REFERENCE_TEMPLATES_FOLDER,
                            threshold=SKELETON_MATCH_THRESHOLD):
    """
    Compare candidate skeleton to ALL .npy templates in templates_folder.
    Return True if the best (lowest) disparity is below threshold.
    """
    cand = np.load(candidate_skeleton_path)

    best_disparity = None

    # Loop over all reference skeleton templates
    for fname in os.listdir(templates_folder):
        if not fname.endswith(".npy"):
            continue

        ref_path = os.path.join(templates_folder, fname)
        ref = np.load(ref_path)

        # Ensure same shape (same number of joints etc.)
        if ref.shape != cand.shape:
            continue

        try:
            _, _, disparity = procrustes(ref, cand)
        except Exception as e:
            print(f"[ERROR] Procrustes failed for {ref_path}: {e}")
            continue

        if best_disparity is None or disparity < best_disparity:
            best_disparity = disparity

    if best_disparity is None:
        print("[WARN] No valid reference skeletons found for matching.")
        return False, None

    is_match = best_disparity < threshold
    return is_match, best_disparity


def normalize_and_sample(points, num_points=NUM_POINTS):
    centroid = np.mean(points, axis=0)
    pts = points - centroid
    furthest_dist = np.max(np.linalg.norm(pts, axis=1))
    if furthest_dist > 0:
        pts /= furthest_dist

    if pts.shape[0] >= num_points:
        idxs = np.random.choice(pts.shape[0], num_points, replace=False)
    else:
        idxs = np.random.choice(pts.shape[0], num_points, replace=True)
    return pts[idxs]


# --- NEW: helper to crop points around skeleton joints --------------------- #
def crop_points_around_skeleton(cluster_pts, skeleton_xyz, radius=0.10):
    """
    Select only points in cluster_pts that lie within `radius` (meters) of
    at least one skeleton joint. This shrinks the bbox toward the mannequin
    body and avoids including far-away objects like the can.
    """
    cluster_pts = np.asarray(cluster_pts)
    skeleton_xyz = np.asarray(skeleton_xyz)

    if skeleton_xyz.ndim != 2 or skeleton_xyz.shape[1] != 3:
        raise ValueError(f"Skeleton array must be (M, 3), got {skeleton_xyz.shape}")

    mask = np.zeros(cluster_pts.shape[0], dtype=bool)
    r2 = radius * radius

    # For each joint, keep points close to it
    for j in range(skeleton_xyz.shape[0]):
        diff = cluster_pts - skeleton_xyz[j]
        d2 = np.sum(diff * diff, axis=1)
        mask |= (d2 <= r2)

    return cluster_pts[mask]
# --------------------------------------------------------------------------- #


def main():
    # Load model
    model = TinyPointNetClassifier(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"[INFO] Loaded model from {MODEL_PATH}")

    # Load scene
    pcd = o3d.io.read_point_cloud(PLY_FILE)
    points = np.asarray(pcd.points)
    if points.shape[0] == 0:
        print("[ERROR] Point cloud is empty.")
        return

    print(f"[INFO] Loaded {points.shape[0]} points from {PLY_FILE}")
    print(f"[DEBUG] Scale: min={points.min(axis=0)}, max={points.max(axis=0)}")

    # DBSCAN segmentation
    clusters = segment_scene_with_dbscan(points)
    print(f"[INFO] DBSCAN found {len(clusters)} valid clusters")

    colors = plt.get_cmap("tab10").colors
    colored_clusters = []
    detected_boxes = []

    yellow_boxes = 0
    green_boxes = 0

    for label, cluster_pts in clusters.items():
        print(f"\n[CLUSTER {label}] size: {cluster_pts.shape[0]} points")

        # Normalize + sample
        pc_sampled = normalize_and_sample(cluster_pts, NUM_POINTS)
        input_tensor = torch.from_numpy(pc_sampled).unsqueeze(0).float().to(DEVICE)
        input_tensor = input_tensor.permute(0, 2, 1)  # (B, 3, N)

        # TinyPointNet inference
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)

        p_back = probs[0, 0].item()
        p_man = probs[0, 1].item()
        pred = int(probs.argmax(dim=1).item())

        print(f"[DEBUG] TinyPointNet pred={pred} (0=background, 1=mannequin), "
              f"P(back)={p_back:.2f}, P(man)={p_man:.2f}")

        # Build point cloud for visualization
        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster_pts)
        cluster_pcd.paint_uniform_color(colors[label % len(colors)])
        colored_clusters.append(cluster_pcd)

        # --- 1) Ignore clusters that are very unlikely to be mannequin --- #
        if p_man < P_MANNEQUIN_MIN_FOR_CHECK:
            print("[INFO] Cluster treated as background (P(man) too low).")
            continue

        # --- 2) Skeleton check (final confirmation / disambiguation) --- #
        skeleton_path = os.path.join(
            SKELETON_FOLDER,
            f"cluster_{label}_skeleton.npy"
        )

        skeleton_matched = False
        disparity = None

        if os.path.exists(skeleton_path):
            skeleton_matched, disparity = is_skeleton_match_multi(
                skeleton_path,
                templates_folder=REFERENCE_TEMPLATES_FOLDER,
                threshold=SKELETON_MATCH_THRESHOLD
            )

            if skeleton_matched:
                print(f"[✓] Skeleton CONFIRMED for cluster {label} "
                      f"(best disparity={disparity:.4f}).")
            else:
                print(f"[✗] Skeleton mismatch for cluster {label} "
                      f"(best disparity={disparity}).")
        else:
            print(f"[!] No skeleton file for cluster {label}: {skeleton_path}")

        # --- 3) Decide whether to draw a bbox & which color --- #
        is_classifier_strong = (p_man >= CONFIDENCE_FOR_YELLOW_BOX)

        # Recognized as mannequin if classifier is strong OR skeleton confirms
        recognized_as_mannequin = is_classifier_strong or skeleton_matched

        if not recognized_as_mannequin:
            print("[INFO] Cluster not recognized as mannequin (no strong evidence).")
            continue

        # --- NEW: choose which points to use for bbox ------------------- #
        # Default: use the whole cluster
        cluster_pts_for_box = cluster_pts

        # If skeleton is confirmed and file exists, crop around skeleton joints
        if skeleton_matched and os.path.exists(skeleton_path):
            try:
                skel_xyz = np.load(skeleton_path)  # (M, 3)
                cropped = crop_points_around_skeleton(
                    cluster_pts, skel_xyz, radius=0.10  # tune radius as needed
                )
                if cropped.shape[0] > 0:
                    print(f"[INFO] Cropped bbox points for cluster {label}: "
                          f"{cropped.shape[0]} / {cluster_pts.shape[0]} points.")
                    cluster_pts_for_box = cropped
                else:
                    print(f"[WARN] Cropping around skeleton for cluster {label} "
                          f"returned 0 points, using full cluster.")
            except Exception as e:
                print(f"[ERROR] Failed to crop around skeleton for cluster {label}: {e}")
        # ---------------------------------------------------------------- #

        # Create AABB from selected points
        aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(cluster_pts_for_box)
        )

        if skeleton_matched:
            # Skeleton-confirmed -> GREEN
            aabb.color = (0.0, 1.0, 0.0)
            green_boxes += 1
            print(f"[INFO] Cluster {label} -> GREEN bbox (skeleton confirmed).")
        else:
            # Classifier-only -> YELLOW
            aabb.color = (1.0, 1.0, 0.0)
            yellow_boxes += 1
            print(f"[INFO] Cluster {label} -> YELLOW bbox (classifier only).")

        detected_boxes.append(aabb)

    print(f"\n[INFO] Final visualization: "
          f"{len(colored_clusters)} clusters, "
          f"{len(detected_boxes)} total bboxes "
          f"({green_boxes} green, {yellow_boxes} yellow).")

    if colored_clusters:
        o3d.visualization.draw_geometries(colored_clusters + detected_boxes)


if __name__ == "__main__":
    main()
