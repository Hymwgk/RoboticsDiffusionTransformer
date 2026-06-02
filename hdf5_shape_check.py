import h5py
import numpy as np

hdf5_path = "/data/datasets/jungong/Sorting-Bullets.hdf5"

with h5py.File(hdf5_path, "r") as f:

    demo = f["data"]["demo_0"]

    print("===== OBS =====")

    for k in demo["obs"].keys():
        data = demo["obs"][k]

        print(
            f"{k:<25} "
            f"shape={data.shape} "
            f"dtype={data.dtype}"
        )

        if len(data.shape) <= 2:
            print(" first sample:")
            print(np.asarray(data[0]))
            print()