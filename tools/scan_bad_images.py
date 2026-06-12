import h5py
import cv2
import numpy as np
from pathlib import Path

DATASET_DIR = Path("/data/rdt_js")
CAM_KEYS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

bad = []

for h5_path in sorted(DATASET_DIR.rglob("*.hdf5")):
    print(f"checking {h5_path}")
    try:
        with h5py.File(h5_path, "r") as f:
            for cam in CAM_KEYS:
                if cam not in f["observations"]["images"]:
                    bad.append((str(h5_path), cam, "missing camera"))
                    continue

                ds = f["observations"]["images"][cam]
                for i in range(len(ds)):
                    buf = ds[i]
                    
                    arr = np.frombuffer(buf, dtype=np.uint8)

                    if len(arr) < 2 or not (arr[-2] == 0xFF and arr[-1] == 0xD9):
                        bad.append((str(h5_path), cam, i, "jpeg missing EOI FF D9"))

                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                    if img is None:
                        bad.append((str(h5_path), cam, i, "imdecode None"))
                    elif img.size == 0:
                        bad.append((str(h5_path), cam, i, "empty image"))

    except Exception as e:
        bad.append((str(h5_path), "file_error", repr(e)))

print("\n========== BAD IMAGES ==========")
for item in bad:
    print(item)

print(f"\nTotal bad items: {len(bad)}")