import re
import json
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--expfolder", type=str)

args = parser.parse_args() 

EXPERIMENT_FOLDER=args.expfolder
JSON_PATH = os.path.join(EXPERIMENT_FOLDER, "experiment.json")
PLOTS_FOLDER = os.path.join(EXPERIMENT_FOLDER, "plots")

def build_time_matrix(json_path: str) -> dict:
    with open(json_path) as f:
        experiment = json.load(f)

    # collect all unique axis values
    n_points_set = set()
    alpha_set    = set()

    for entry in experiment["results"]["mtscan"]:
        n_points_set.add(entry["n_points"])
        alpha_set.add(entry["parameters"]["alpha"])

    n_points_vals = sorted(n_points_set)
    alpha_vals    = sorted(alpha_set)

    n_pts_idx = {v: i for i, v in enumerate(n_points_vals)}
    alpha_idx = {v: i for i, v in enumerate(alpha_vals)}

    matrix = np.full((len(n_points_vals), len(alpha_vals)), np.nan)

    for entry in experiment["results"]["mtscan"]:
        timers = entry.get("timers")
        if timers is None:
            continue
        i = n_pts_idx[entry["n_points"]]
        j = alpha_idx[entry["parameters"]["alpha"]]
        matrix[i, j] = timers.get("TOTAL", np.nan)

    return {
        "matrix":   matrix,
        "n_points": n_points_vals,
        "alphas":   alpha_vals,
    }



def plot_time_vs_npoints(
    time_data,
    alpha_range,
    title,
    save_path
):
    matrix    = time_data["matrix"]
    n_points  = time_data["n_points"]
    alphas    = time_data["alphas"]

    alphas_to_plot = alpha_range if alpha_range is not None else alphas
    # keep only alphas that actually exist in the data
    alphas_to_plot = [a for a in alphas_to_plot if a in alphas]

    colors = cm.plasma(np.linspace(0.1, 0.9, len(alphas_to_plot)))

    fig, ax = plt.subplots(figsize=(9, 5))

    for color, alpha in zip(colors, alphas_to_plot):
        j    = alphas.index(alpha)
        col  = matrix[:, j]
        mask = ~np.isnan(col)
        ax.plot(
            np.array(n_points)[mask],
            col[mask],
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=color,
            label=f"α={alpha}",
        )

    ax.set_xlabel("n_points")
    ax.set_ylabel("TOTAL time (s)")
    ax.set_title(title)
    ax.legend(title="alpha", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xscale("log")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()

    return ax



def plot_time_vs_alpha(
    time_data,
    npoints_range,
    title,
    save_path
):
    matrix   = time_data["matrix"]
    n_points = time_data["n_points"]
    alphas   = time_data["alphas"]

    npts_to_plot = npoints_range if npoints_range is not None else n_points
    npts_to_plot = [n for n in npts_to_plot if n in n_points]

    colors = cm.viridis(np.linspace(0.1, 0.9, len(npts_to_plot)))

    fig, ax = plt.subplots(figsize=(9, 5))

    for color, npts in zip(colors, npts_to_plot):
        i    = n_points.index(npts)
        row  = matrix[i, :]
        mask = ~np.isnan(row)
        ax.plot(
            np.array(alphas)[mask],
            row[mask],
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=color,
            label=f"n={npts:,}",
        )

    ax.set_xlabel("alpha")
    ax.set_ylabel("TOTAL time (s)")
    ax.set_title(title)
    ax.legend(title="n_points", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()

    return ax


if __name__ == "__main__":

    time_data = build_time_matrix(JSON_PATH)

    #save_path=os.path.join(PLOTS_FOLDER, "time_vs_n_points.png")
    save_path=f"{PLOTS_FOLDER}/time_vs_n_points.png"

    alpha_range=[25, 50, 75, 100]
    title="Runtime vs number of points"

    plot_time_vs_npoints(
            time_data,
            alpha_range,
            title,
            save_path,
        )

    npoints_range= [100_000, 200_000, 500_000, 1_000_000]
    title="Runtime vs alpha"
    #save_path=os.path.join(PLOTS_FOLDER, "time_vs_alpha.png"),
    save_path=f"{PLOTS_FOLDER}/time_vs_alpha.png"

    plot_time_vs_alpha(
            time_data,
            npoints_range,
            title,
            save_path
        )

