import os
import numpy as np
import imageio.v2 as imageio

#swap tiff to npy
# This script is present because of issues I faced in WSL

src = "datasets/UseGeo/Dataset-1/depth_maps"
dst = "datasets/UseGeo/Dataset-1/depth_npy"
os.makedirs(dst, exist_ok=True)

for f in os.listdir(src):
    if f.endswith(".tiff") or f.endswith(".tif"):
        arr = imageio.imread(os.path.join(src, f))
        np.save(os.path.join(dst, f.replace(".tiff", ".npy").replace(".tif", ".npy")), arr)
