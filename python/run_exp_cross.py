import os
import json
import shutil
import numpy as np
import hdbscan
import mt_scan
from datetime import datetime
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--overwrite",   action="store_true")
parser.add_argument("-i", "--id",          type=str, default="exp_cross_001")
parser.add_argument("-r", "--resolution",  type=int, default=512)
parser.add_argument("-t", "--testmode",    action="store_true")
args = parser.parse_args()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

EXPERIMENT_ID   = args.id
DATASET_FOLDER  = "/workspace/data/sample" if args.testmode else "/workspace/data/synthetic-1"
RESULTS_FOLDER  = os.path.join(DATASET_FOLDER, "results")
EXPERIMENT_FOLDER = os.path.join(RESULTS_FOLDER, EXPERIMENT_ID)
PLOTS_FOLDER    = os.path.join(EXPERIMENT_FOLDER, "plots")
LABELS_FOLDER   = os.path.join(EXPERIMENT_FOLDER, "_labels_tmp")

KERNEL       = "gaussian"
RESOLUTION   = args.resolution
ALPHA_RANGE  = list(range(10, 201, 10))
MINPTS_RANGE = [5, 10, 15, 20, 25, 30, 40, 50]

if os.path.exists(EXPERIMENT_FOLDER):
    if args.overwrite:
        shutil.rmtree(EXPERIMENT_FOLDER)
        print(f"Overwriting existing experiment '{EXPERIMENT_ID}'")
    else:
        raise FileExistsError(f"Experiment '{EXPERIMENT_ID}' already exists. Use -o to overwrite.")

os.makedirs(PLOTS_FOLDER)

csv_files = sorted([f for f in os.listdir(DATASET_FOLDER) if f.endswith(".csv")])

print("=" * 50)
print(f"  EXPERIMENT {EXPERIMENT_ID}  —  cross-ARI")
print("=" * 50)
print(f"  Dataset folder : {DATASET_FOLDER}")
print(f"  Resolution     : {RESOLUTION}")
print(f"  Alpha range    : {ALPHA_RANGE}")
print(f"  MinPts range   : {MINPTS_RANGE}")
print(f"  Datasets       : {csv_files}")
print("=" * 50)

experiment = {
    "experiment_id": EXPERIMENT_ID,
    "timestamp":     datetime.now().isoformat(),
    "parameters": {
        "kernel":       KERNEL,
        "resolution":   RESOLUTION,
        "alpha_range":  ALPHA_RANGE,
        "minpts_range": MINPTS_RANGE,
        "dataset_list": csv_files,
    },
    "results": []
}

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
for csv_file in csv_files:
    dataset_name = csv_file.replace(".csv", "")
    print(f"\n── {dataset_name} ──")

    try:
        data      = np.loadtxt(os.path.join(DATASET_FOLDER, csv_file), delimiter=",", skiprows=1)
        points    = data[:, :2].astype(np.float32)
        labels_gt = data[:, 2].astype(int)

        # ── compute & cache all mtscan labels ───────────────────────────
        print(f"  Computing mt_scan labels for {len(ALPHA_RANGE)} alpha values...")
        mtscan_labels = {}
        for alpha in ALPHA_RANGE:
            labels = mt_scan.compute_labels(
                points, alpha=alpha, target_res=RESOLUTION, kernel=KERNEL
            )
            mtscan_labels[alpha] = labels

        # ── compute & cache all hdbscan labels ──────────────────────────
        print(f"  Computing hdbscan labels for {len(MINPTS_RANGE)} minpts values...")
        hdbscan_labels = {}
        for minpts in MINPTS_RANGE:
            labels = hdbscan.HDBSCAN(min_cluster_size=minpts).fit_predict(points)
            hdbscan_labels[minpts] = labels

        # ── cross ARI matrix  (rows=alpha, cols=minpts) ─────────────────
        print(f"  Computing {len(ALPHA_RANGE)}×{len(MINPTS_RANGE)} ARI matrix...")
        ari_matrix = np.zeros((len(ALPHA_RANGE), len(MINPTS_RANGE)))

        ari_grid = []   # list of dicts for JSON
        for i, alpha in enumerate(ALPHA_RANGE):
            row = []
            for j, minpts in enumerate(MINPTS_RANGE):
                ari = adjusted_rand_score(
                    mtscan_labels[alpha],
                    hdbscan_labels[minpts]
                )
                ari_matrix[i, j] = ari
                row.append({"alpha": alpha, "minpts": minpts, "ari": round(float(ari), 6)})
            ari_grid.append(row)

        # ── plot heatmap ─────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(max(8, len(MINPTS_RANGE) * 0.9),
                                        max(6, len(ALPHA_RANGE) * 0.4)))

        im = ax.imshow(
            ari_matrix,
            aspect="auto",
            cmap="RdYlGn",
            vmin=-0.1,
            vmax=1.0,
            origin="upper",
        )

        ax.set_xticks(range(len(MINPTS_RANGE)))
        ax.set_xticklabels(MINPTS_RANGE)
        ax.set_yticks(range(len(ALPHA_RANGE)))
        ax.set_yticklabels(ALPHA_RANGE)

        ax.set_xlabel("HDBSCAN  min_pts")
        ax.set_ylabel("MT-SCAN  alpha")
        ax.set_title(f"{dataset_name}  —  cross ARI  (mt_scan vs hdbscan)")

        # annotate cells if grid is small enough
        if len(ALPHA_RANGE) * len(MINPTS_RANGE) <= 200:
            for i in range(len(ALPHA_RANGE)):
                for j in range(len(MINPTS_RANGE)):
                    v = ari_matrix[i, j]
                    color = "black" if 0.2 < v < 0.8 else "white"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=7, color=color)

        plt.colorbar(im, ax=ax, label="ARI")
        plt.tight_layout()

        plot_path = os.path.join(PLOTS_FOLDER, f"{dataset_name}_cross_ari.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved plot → {plot_path}")

        # ── best pair ────────────────────────────────────────────────────
        best_idx   = np.unravel_index(np.argmax(ari_matrix), ari_matrix.shape)
        best_alpha  = ALPHA_RANGE[best_idx[0]]
        best_minpts = MINPTS_RANGE[best_idx[1]]
        best_ari    = float(ari_matrix[best_idx])

        experiment["results"].append({
            "dataset_name": dataset_name,
            "n_points":     len(points),
            "best_pair": {
                "alpha":  best_alpha,
                "minpts": best_minpts,
                "ari":    round(best_ari, 6),
            },
            "ari_grid": ari_grid,
            "plot":     f"plots/{dataset_name}_cross_ari.png",
        })

        print(f"  Best pair: alpha={best_alpha}, minpts={best_minpts}, ARI={best_ari:.4f}")

    except Exception as e:
        print(f"  ERROR on {dataset_name}: {e} — skipping")
        experiment["results"].append({"dataset_name": dataset_name, "error": str(e)})


# ─────────────────────────────────────────────
# SAVE JSON
# ─────────────────────────────────────────────
json_path = os.path.join(EXPERIMENT_FOLDER, "experiment.json")
with open(json_path, "w") as f:
    json.dump(experiment, f, indent=2)

print(f"\nResults saved → {json_path}")