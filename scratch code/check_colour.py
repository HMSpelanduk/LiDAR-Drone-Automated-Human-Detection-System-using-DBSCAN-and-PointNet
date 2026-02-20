# check_colour.py
import open3d as o3d
import numpy as np
import os

PLY_FILE = r"data\more_test.ply"   # <- change this to your actual file

def main():
    print(f"[INFO] Exists? {os.path.exists(PLY_FILE)}")
    print(f"[INFO] Loading: {PLY_FILE}")
    pcd = o3d.io.read_point_cloud(PLY_FILE)
    print(f"[INFO] Number of points: {len(pcd.points)}")
    print(f"[INFO] pcd.has_colors(): {pcd.has_colors()}")

    if not pcd.has_colors():
        print("[WARN] No color data stored in this PLY (only XYZ or scalar fields).")
        return

    colors = np.asarray(pcd.colors)
    print(f"[INFO] First 5 colors (R,G,B in [0,1]):\n{colors[:5]}")

if __name__ == "__main__":
    main()
