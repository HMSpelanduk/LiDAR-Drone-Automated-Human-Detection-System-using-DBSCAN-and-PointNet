import numpy as np
import os
import xml.etree.ElementTree as ET

# === Fixed input file ===
pp_file = "../cluster_1.pp"

if not os.path.exists(pp_file):
    print(f"[ERROR] File not found: {pp_file}")
    exit()

# === Parse MeshLab .pp file ===
try:
    tree = ET.parse(pp_file)
    root = tree.getroot()

    points = []
    for point in root.findall(".//point"):
        x = float(point.attrib['x'])
        y = float(point.attrib['y'])
        z = float(point.attrib['z'])
        points.append([x, y, z])

    points = np.array(points)
    print(f"[INFO] Loaded {len(points)} skeleton points from {pp_file}")

    # === Save as .npy ===
    os.makedirs("../manual_skeletons", exist_ok=True)
    out_path = "../manual_skeletons/cluster_1_skeleton.npy"
    np.save(out_path, points)
    print(f"[✓] Saved to {out_path}")

except Exception as e:
    print(f"[ERROR] Failed to parse or save: {e}")
