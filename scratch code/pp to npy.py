import numpy as np
import os


def convert_pp_to_npy(pp_path, npy_path):
    points = []

    with open(pp_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('point'):
                parts = line.split()
                if len(parts) >= 4:
                    x, y, z = map(float, parts[1:4])
                    points.append([x, y, z])

    points = np.array(points, dtype=np.float32)
    np.save(npy_path, points)
    print(f"[✓] Saved {len(points)} joints to: {npy_path}")


# === MAIN EXECUTION ===
if __name__ == "__main__":
    pp_file = r"/man_1.pp"
    output_npy = "data_npy/mannequin/man_1_skeleton.npy"

    os.makedirs(os.path.dirname(output_npy), exist_ok=True)
    convert_pp_to_npy(pp_file, output_npy)
