import mt_scan
import pickle
import numpy as np
from pathlib import Path
import json
import time
import hdbscan
import argparse
import os
import sys



#dataset = args.dataset
pkl_file = f"/worksapce/data/blobs_grid_1000.pkl"

with open(pkl_file, "rb") as f:
    points = pickle.load(f)
points = np.array(points)

points = points[:,:2].astype(np.float32)


#points = points[:, :2].astype(np.float32).ravel()
n_points=points.shape[0]

t_start = time.perf_counter()

labels=mt_scan.compute_labels(points, 512, 100)

t_mtscan = time.perf_counter()


print(f"{' -- MTSCAN':<70} | {t_mtscan - t_start:.4f}s", file=sys.stderr)

import matplotlib.pyplot as plt
plt.figure()

noise = labels == -1

plt.scatter(points[~noise, 0], points[~noise, 1],
            c=labels[~noise], s=15)
plt.scatter(points[noise, 0], points[noise, 1],
            c='gray', s=30, alpha=0.7)

plt.show()
