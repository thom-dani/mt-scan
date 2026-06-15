import os
import json
import numpy as np
import hdbscan
import mt_scan
from datetime import datetime
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--overwrite", action="store_true",
                    help="Overwrite an existing experiment with the same ID")
args = parser.parse_args()

# ─────────────────────────────────────────────
# EXPERIMENT PARAMETERS — fill these in
# ─────────────────────────────────────────────

EXPERIMENT_ID   = "exp_003"
DATASET_FOLDER  = "/workspace/data/synthetic-1"
RESULTS_FOLDER  = os.path.join(DATASET_FOLDER, "results")
PLOTS_FOLDER    = os.path.join(RESULTS_FOLDER, EXPERIMENT_ID, "plots")



experiment_folder = os.path.join(RESULTS_FOLDER, EXPERIMENT_ID)

if os.path.exists(experiment_folder):
    if args.overwrite:
        import shutil
        shutil.rmtree(experiment_folder)
        print(f"Overwriting existing experiment '{EXPERIMENT_ID}'")
    else:
        raise FileExistsError(f"There is already an experiment with id {EXPERIMENT_ID} for the dataset {DATASET_FOLDER}.")

os.makedirs(PLOTS_FOLDER)


KERNEL          = "gaussian"       
RESOLUTION      = 512
DEVICE          = "cpu"

ALPHA_RANGE     = list(range(10, 201, 10))
MINPTS_RANGE    = [5, 10, 15, 20, 25, 30, 40, 50]

csv_files = sorted([f for f in os.listdir(DATASET_FOLDER) if f.endswith(".csv")])

print("=" * 50)
print(f"  EXPERIMENT {EXPERIMENT_ID}")
print("=" * 50)
print(f"  Dataset folder : {DATASET_FOLDER}")
print(f"  Results folder : {experiment_folder}")
print(f"  Device         : {DEVICE}")
print(f"  Kernel         : {KERNEL}")
print(f"  Resolution     : {RESOLUTION}")
print(f"  Alpha range    : {list(ALPHA_RANGE)}")
print(f"  MinPts range   : {MINPTS_RANGE}")
print(f"  Datasets       : {csv_files}")
print("=" * 50)

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
os.makedirs(PLOTS_FOLDER, exist_ok=True)

experiment = {
    "experiment_id": EXPERIMENT_ID,
    "timestamp": datetime.now().isoformat(),
    "parameters": {
        "kernel":       KERNEL,
        "resolution":   RESOLUTION,
        "device":       DEVICE,
        "alpha_range":  ALPHA_RANGE,
        "minpts_range": MINPTS_RANGE,
        "dataset_list": []
    },
    "results": []
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def compute_scores(labels_gt, labels_pred):
    return {
        "ari": adjusted_rand_score(labels_gt, labels_pred),
        "nmi": normalized_mutual_info_score(labels_gt, labels_pred),
        "ami": adjusted_mutual_info_score(labels_gt, labels_pred)
    }

def find_best(sweep, metric):
    best = max(sweep, key=lambda x: x[metric])
    return best[metric], best["param"]

def save_plot(dataset_name, min_pts, alpha, indicator):
    data           = np.loadtxt(os.path.join(DATASET_FOLDER, dataset_name + ".csv"), delimiter=",", skiprows=1)
    points         = data[:, :2].astype(np.float32)
    labels_gt      = data[:, 2].astype(int)
    labels_mtscan  = mt_scan.compute_labels(points, alpha, RESOLUTION)
    clusterer      = hdbscan.HDBSCAN(min_cluster_size=min_pts)
    labels_hdbscan = clusterer.fit_predict(points)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- ground truth ---
    for l in [l for l in np.unique(labels_gt) if l != -1]:
        axes[0].scatter(points[labels_gt == l, 0], points[labels_gt == l, 1], s=15)
    if (labels_gt == -1).any():
        axes[0].scatter(points[labels_gt == -1, 0], points[labels_gt == -1, 1], c='gray', s=15, alpha=0.5)
    axes[0].set_title("ground_truth")
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # --- mt_scan ---
    for l in [l for l in np.unique(labels_mtscan) if l != -1]:
        axes[1].scatter(points[labels_mtscan == l, 0], points[labels_mtscan == l, 1], s=15)
    if (labels_mtscan == -1).any():
        axes[1].scatter(points[labels_mtscan == -1, 0], points[labels_mtscan == -1, 1], c='gray', s=15, alpha=0.5)
    axes[1].set_title(f"mt_scan {alpha}")
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # --- hdbscan ---
    for l in [l for l in np.unique(labels_hdbscan) if l != -1]:
        axes[2].scatter(points[labels_hdbscan == l, 0], points[labels_hdbscan == l, 1], s=15)
    if (labels_hdbscan == -1).any():
        axes[2].scatter(points[labels_hdbscan == -1, 0], points[labels_hdbscan == -1, 1], c='gray', s=15, alpha=0.5)
    axes[2].set_title(f"hdbscan {min_pts}")
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    path = os.path.join(PLOTS_FOLDER, f"{dataset_name}_best_{indicator}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

experiment["parameters"]["dataset_list"] = csv_files

for csv_file in csv_files:
    dataset_name = csv_file.replace(".csv", "")
    print(f"\n── {dataset_name} ──")
    try:
        data      = np.loadtxt(os.path.join(DATASET_FOLDER, csv_file), delimiter=",", skiprows=1)
        points    = data[:, :2].astype(np.float32)
        labels_gt = data[:, 2].astype(int)

        # ── MTSCAN ──────────────────────────────
        mtscan_sweep = []
        for alpha in ALPHA_RANGE:
            labels_pred     = mt_scan.compute_labels(points, alpha, RESOLUTION)
            scores          = compute_scores(labels_gt, labels_pred)
            scores["param"] = alpha
            mtscan_sweep.append(scores)

        best_ari_mtscan, best_alpha_ari = find_best(mtscan_sweep, "ari")
        best_nmi_mtscan, best_alpha_nmi = find_best(mtscan_sweep, "nmi")
        best_ami_mtscan, best_alpha_ami = find_best(mtscan_sweep, "ami")

        # ── HDBSCAN ─────────────────────────────
        hdbscan_sweep = []
        for minpts in MINPTS_RANGE:
            clusterer       = hdbscan.HDBSCAN(min_cluster_size=minpts)
            labels_pred     = clusterer.fit_predict(points)
            scores          = compute_scores(labels_gt, labels_pred)
            scores["param"] = minpts
            hdbscan_sweep.append(scores)

        best_ari_hdbscan, best_minpts_ari = find_best(hdbscan_sweep, "ari")
        best_nmi_hdbscan, best_minpts_nmi = find_best(hdbscan_sweep, "nmi")
        best_ami_hdbscan, best_minpts_ami = find_best(hdbscan_sweep, "ami")

        save_plot(dataset_name, best_minpts_ari, best_alpha_ari, "ari")

        plots = {
            "ari":        f"plots/{dataset_name}_best_ari.png",
            "nmi":        f"plots/{dataset_name}_best_nmi.png",
            "ami":        f"plots/{dataset_name}_best_ami.png",
        }

        experiment["results"].append({
            "dataset_name": dataset_name,
            "n_points":     len(points),
            "mtscan": {
                "best_ari":       best_ari_mtscan,
                "best_alpha_ari": best_alpha_ari,
                "best_nmi":       best_nmi_mtscan,
                "best_alpha_nmi": best_alpha_nmi,
                "best_ami":       best_ami_mtscan,
                "best_alpha_ami": best_alpha_ami,
                "sweep":          mtscan_sweep
            },
            "hdbscan": {
                "best_ari":        best_ari_hdbscan,
                "best_minpts_ari": best_minpts_ari,
                "best_nmi":        best_nmi_hdbscan,
                "best_minpts_nmi": best_minpts_nmi,
                "best_ami":        best_ami_hdbscan,
                "best_minpts_ami": best_minpts_ami,
                "sweep":           hdbscan_sweep
            },
            "plots": plots
        })

    except Exception as e:
        print(f"  ERROR on {dataset_name}: {e} — skipping")
        continue
# ─────────────────────────────────────────────
# SAVE JSON
# ─────────────────────────────────────────────
json_path = os.path.join(RESULTS_FOLDER, EXPERIMENT_ID, "experiment.json")
os.makedirs(os.path.dirname(json_path), exist_ok=True)
with open(json_path, "w") as f:
    json.dump(experiment, f, indent=2)

print(f"\nResults saved to {json_path}")