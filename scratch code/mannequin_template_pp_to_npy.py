import numpy as np
import os
import xml.etree.ElementTree as ET

# Load and parse the MeshLab .pp file
tree = ET.parse("../man_1.pp")
root = tree.getroot()

# Extract 3D points from <point> tags
points = []
for point in root.findall(".//point"):
    x = float(point.attrib['x'])
    y = float(point.attrib['y'])
    z = float(point.attrib['z'])
    points.append([x, y, z])

points = np.array(points)
print(f"[INFO] Loaded {len(points)} skeleton points from man_1.pp")

# Save as mannequin_template.npy
os.makedirs("../reference_templates", exist_ok=True)
np.save("../reference_templates/mannequin_template.npy", points)
print("[✓] Saved to reference_templates/mannequin_template.npy")


# === Input .pp filename ===
# Usage: python pp_to_npy.py cluster_3.pp
if len(sys.argv) < 2:
    print("Usage: python pp_to_npy.py <filename.pp>")
    exit()

pp_file = sys.argv[1]
if not os.path.exists(pp_file):
    print(f"[ERROR] File not found: {pp_file}")
    exit()

