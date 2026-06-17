import re
import json
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--expfolder", type=str)

args = parser.parse_args() 

EXPERIMENT_FOLDER=args.expfolder
JSON_PATH = os.path.join(EXPERIMENT_FOLDER, "experiment.json")
PLOTS_FOLDER = os.path.join(EXPERIMENT_FOLDER, "plots")


def build_time_matrix_bars(json_path: str) -> dict:
    with open(json_path) as f:
        experiment = json.load(f)

    n_points_set = set()
    alpha_set    = set()

    for entry in experiment["results"]["mtscan"]:
        n_points_set.add(entry["n_points"])
        alpha_set.add(entry["parameters"]["alpha"])

    n_points_vals = sorted(n_points_set)
    alpha_vals    = sorted(alpha_set)

    n_pts_idx = {v: i for i, v in enumerate(n_points_vals)}
    alpha_idx = {v: i for i, v in enumerate(alpha_vals)}

    matrix = np.full((len(n_points_vals), len(alpha_vals), 4), np.nan)

    for entry in experiment["results"]["mtscan"]:
        timers=entry.get("timers", {})
        time_resample=timers.get("resample")
        time_tree=timers.get("clusterTree")
        time_labels=timers.get("computeLabels")
        time_total=timers.get("total")
        i = n_pts_idx[entry["n_points"]]
        j = alpha_idx[entry["parameters"]["alpha"]]
        matrix[i, j] = [time_resample, time_tree, time_labels, time_total]

    return {
        "matrix":   matrix,
        "n_points": n_points_vals,
        "alphas":   alpha_vals,
    }

def build_time_matrix_alpha(json_path: str) -> dict:
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
        time_s=entry.get("time_s")
        i = n_pts_idx[entry["n_points"]]
        j = alpha_idx[entry["parameters"]["alpha"]]
        matrix[i, j] = time_s

    return {
        "matrix":   matrix,
        "n_points": n_points_vals,
        "alphas":   alpha_vals,
    }

def build_time_matrix_minpts(json_path: str) -> dict:
    with open(json_path) as f:
        experiment = json.load(f)

    # collect all unique axis values
    n_points_set = set()
    minpts_set    = set()

    for entry in experiment["results"]["hdbscan"]:
        n_points_set.add(entry["n_points"])
        minpts_set.add(entry["parameters"]["minpts"])

    n_points_vals = sorted(n_points_set)
    minpts_vals    = sorted(minpts_set)

    n_pts_idx = {v: i for i, v in enumerate(n_points_vals)}
    minpts_idx = {v: i for i, v in enumerate(minpts_vals)}

    matrix = np.full((len(n_points_vals), len(minpts_vals)), np.nan)

    for entry in experiment["results"]["hdbscan"]:
        time_s=entry.get("time_s")
        i = n_pts_idx[entry["n_points"]]
        j = minpts_idx[entry["parameters"]["minpts"]]
        matrix[i, j] = time_s

    return {
        "matrix":   matrix,
        "n_points": n_points_vals,
        "minpts":   minpts_vals,
    }

def build_time_matrix_cuml(json_path: str) -> dict:
    with open(json_path) as f:
        experiment = json.load(f)

    # collect all unique axis values
    n_points_set = set()
    minpts_set    = set()

    for entry in experiment["results"]["hdbscan_cuml"]:
        n_points_set.add(entry["n_points"])
        minpts_set.add(entry["parameters"]["minpts"])

    n_points_vals = sorted(n_points_set)
    minpts_vals    = sorted(minpts_set)

    n_pts_idx = {v: i for i, v in enumerate(n_points_vals)}
    minpts_idx = {v: i for i, v in enumerate(minpts_vals)}

    matrix = np.full((len(n_points_vals), len(minpts_vals)), np.nan)

    for entry in experiment["results"]["hdbscan_cuml"]:
        time_s=entry.get("time_s")
        i = n_pts_idx[entry["n_points"]]
        j = minpts_idx[entry["parameters"]["minpts"]]
        matrix[i, j] = np.nan if time_s is None else time_s


    return {
        "matrix":   matrix,
        "n_points": n_points_vals,
        "minpts":   minpts_vals,
    }

def save_plot(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)

def plot_time_vs_npoints_alpha(
    time_data,
    alpha_range,
    title,
    ax
):
    matrix    = time_data["matrix"]
    n_points  = time_data["n_points"]
    alphas    = time_data["alphas"]

    alphas_to_plot = alpha_range if alpha_range is not None else alphas
    # keep only alphas that actually exist in the data
    alphas_to_plot = [a for a in alphas_to_plot if a in alphas]

    colors = cm.plasma(np.linspace(0.1, 0.9, len(alphas_to_plot)))

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
    ax.grid(True, linestyle="--", alpha=0.4)

    return ax



def plot_time_vs_alpha(
    time_data,
    npoints_range,
    title,
    ax
):
    matrix   = time_data["matrix"]
    n_points = time_data["n_points"]
    alphas   = time_data["alphas"]

    npts_to_plot = npoints_range if npoints_range is not None else n_points
    npts_to_plot = [n for n in npts_to_plot if n in n_points]

    colors = cm.viridis(np.linspace(0.1, 0.9, len(npts_to_plot)))

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

    return ax

def plot_time_vs_npoints_minpts(
    time_data,
    minpts_range,
    title,
    ax
):
    matrix    = time_data["matrix"]
    n_points  = time_data["n_points"]
    minptss    = time_data["minpts"]

    minpts_to_plot = minpts_range if minpts_range is not None else minpts
    minpts_to_plot = [a for a in minpts_to_plot if a in minptss]

    colors = cm.plasma(np.linspace(0.1, 0.9, len(minpts_to_plot)))

    for color, minpts in zip(colors, minpts_to_plot):
        j    = minptss.index(minpts)
        col  = matrix[:, j]
        mask = ~np.isnan(col)
        ax.plot(
            np.array(n_points)[mask],
            col[mask],
            marker="s",
            linewidth=1.8,
            markersize=4,
            color="green",
            linestyle="--",
            label=f"minpts={minpts}",
        )

    ax.set_xlabel("n_points")
    ax.set_ylabel("TOTAL time (s)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)

    return ax

def plot_time_vs_npoints_cuml(
    time_data,
    minpts_range,
    title,
    ax
):
    matrix    = time_data["matrix"]
    n_points  = time_data["n_points"]
    minptss    = time_data["minpts"]

    minpts_to_plot = minpts_range if minpts_range is not None else minpts
    minpts_to_plot = [a for a in minpts_to_plot if a in minptss]

    colors = cm.plasma(np.linspace(0.1, 0.9, len(minpts_to_plot)))

    for color, minpts in zip(colors, minpts_to_plot):
        j    = minptss.index(minpts)
        col  = matrix[:, j]
        mask = ~np.isnan(col)
        ax.plot(
            np.array(n_points)[mask],
            col[mask],
            marker="s",
            linewidth=1.8,
            markersize=4,
            color="red",
            linestyle="--",
            label=f"minpts(cuML)={minpts}",
        )

    ax.set_xlabel("n_points")
    ax.set_ylabel("TOTAL time (s)")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.4)

    return ax



def plot_time_vs_minpts(
    time_data,
    npoints_range,
    title,
    ax
):
    matrix   = time_data["matrix"]
    n_points = time_data["n_points"]
    minpts   = time_data["minpts"]

    npts_to_plot = npoints_range if npoints_range is not None else n_points
    npts_to_plot = [n for n in npts_to_plot if n in n_points]

    colors = cm.viridis(np.linspace(0.1, 0.9, len(npts_to_plot)))

    for color, npts in zip(colors, npts_to_plot):
        i    = n_points.index(npts)
        row  = matrix[i, :]
        mask = ~np.isnan(row)
        ax.plot(
            np.array(minpts)[mask],
            row[mask],
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=color,
            label=f"n={npts:,}",
        )

    ax.set_xlabel("minpts")
    ax.set_ylabel("TOTAL time (s)")
    ax.set_title(title)
    ax.legend(title="n_points", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()

    return ax

def plot_bars(time_data, alpha, title, ax):
    matrix   = time_data["matrix"]
    n_points = time_data["n_points"]
    alphas   = time_data["alphas"]

    j = alphas.index(alpha)

    labels       = ["resample", "clusterTree", "computeLabels"]
    colors       = ["#4C72B0", "#DD8452", "#55A868"]
    bottom       = np.zeros(len(n_points))

    for k, (label, color) in enumerate(zip(labels, colors)):
        values = matrix[:, j, k]
        ax.bar(range(len(n_points)), values, bottom=bottom, label=label, color=color)
        bottom += np.nan_to_num(values)

    ax.set_xticks(range(len(n_points)))
    ax.set_xticklabels([str(n) for n in n_points], rotation=45, ha="right")
    ax.set_xlabel("n_points")
    ax.set_ylabel("time (s)")
    ax.set_title(title)
    ax.legend()

def plot_bars_alpha_range(time_data, alphas, title, ax):
    matrix     = time_data["matrix"]
    n_points   = time_data["n_points"]
    all_alphas = time_data["alphas"]

    n_pts      = len(n_points)
    n_alphas   = len(alphas)
    bar_width  = 0.8 / n_alphas
    alpha_idxs = [all_alphas.index(a) for a in alphas]

    # One color per alpha
    cmap        = plt.cm.viridis
    alpha_colors = [cmap(i / max(n_alphas - 1, 1)) for i in range(n_alphas)]

    for a_pos, (a_val, j, color) in enumerate(zip(alphas, alpha_idxs, alpha_colors)):
        x      = np.arange(n_pts) + a_pos * bar_width
        values = matrix[:, j, -1]  # last component only

        ax.bar(x, values, width=bar_width * 0.9,  # 0.9 leaves a small gap between bars
               color=color, label=str(a_val))

    # n_points group labels centered under each group
    group_centers = np.arange(n_pts) + (n_alphas - 1) * bar_width / 2
    ax.set_xticks(group_centers)
    ax.set_xticklabels([str(n) for n in n_points], rotation=45, ha="right")

    ax.set_xlabel("n_points")
    ax.set_ylabel("time (s)")
    ax.set_title(title)
    ax.legend(title="alpha")

if __name__ == "__main__":

    time_data_mtscan = build_time_matrix_alpha(JSON_PATH)
    time_data_hdbscan = build_time_matrix_minpts(JSON_PATH)
    time_data_cuml = build_time_matrix_cuml(JSON_PATH)
    time_data_bars = build_time_matrix_bars(JSON_PATH)

    #save_path=os.path.join(PLOTS_FOLDER, "time_vs_n_points.png")
    save_path_1=f"{PLOTS_FOLDER}/time_vs_n_points.png"

    fig1, ax1 = plt.subplots()

    alpha_range=[10, 20, 50]#, 100]
    min_pts_range=[20]
    title="Runtime vs number of points"

    plot_time_vs_npoints_alpha(
            time_data_mtscan,
            alpha_range,
            title,
            ax1
        )

    plot_time_vs_npoints_minpts(
            time_data_hdbscan,
            min_pts_range,
            title,
            ax1
        )
    
    plot_time_vs_npoints_cuml(
            time_data_cuml,
            min_pts_range,
            title,
            ax1
        )
    
    ax1.legend(title="parameters", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    
    ax1.set_yscale("log")   # set scale FIRST
    save_plot(fig1, save_path_1)

    """
    fig2, ax2 = plt.subplots()

    npoints_range= [100_000, 200_000, 500_000, 1_000_000, 1_500_000, 2_000_000]
    title="Runtime vs alpha"
    #save_path=os.path.join(PLOTS_FOLDER, "time_vs_alpha.png"),
    save_path_2=f"{PLOTS_FOLDER}/time_vs_alpha.png"

    plot_time_vs_alpha(
            time_data_mtscan,
            npoints_range,
            title,
            ax2
        )

    save_plot(fig2, save_path_2)

    """


    fig3, ax3 = plt.subplots()

    alpha=10
    title=f"Runtime vs number of points for different alphas"

    save_path_3=f"{PLOTS_FOLDER}/bar_plot_alphas.png"
    plot_bars_alpha_range(
            time_data_bars,
            alpha_range,
            title,
            ax3
        )

    save_plot(fig3, save_path_3)

    fig4, ax4 = plt.subplots()

    alpha=10
    title=f"Runtime vs number of points for alpha = 10"

    save_path_4=f"{PLOTS_FOLDER}/bar_plot_a_10.png"
    plot_bars(
            time_data_bars,
            alpha,
            title,
            ax4
        )

    save_plot(fig4, save_path_4)

    fig5, ax5 = plt.subplots()

    alpha=50
    title=f"Runtime vs number of points for alpha = 50"

    save_path_5=f"{PLOTS_FOLDER}/bar_plot_a_50.png"
    plot_bars(
            time_data_bars,
            alpha,
            title,
            ax5
        )

    save_plot(fig5, save_path_5)


