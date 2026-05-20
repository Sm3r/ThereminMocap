# 2D time-series of X/Y/Z positions for key ZED hand joints
# (wrist, index MCP, pinky MCP) across frames.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import glob
import re

import matplotlib.pyplot as plt
import pandas as pd

from config import config


SIGNIFICANT_JOINTS = [0, 5, 17]
JOINT_LABELS = ["Wrist (0)", "Index MCP (5)", "Pinky MCP (17)"]


def _plot_joint(ax, df, hand, joint_idx):
    x_col = f"{hand}_{joint_idx:02d}_X"
    y_col = f"{hand}_{joint_idx:02d}_Y"
    z_col = f"{hand}_{joint_idx:02d}_Z"
    valid = False
    if x_col in df.columns:
        ax.plot(df["Frame"], df[x_col].values, label="X", linewidth=0.7, alpha=0.8)
        valid = True
    if y_col in df.columns:
        ax.plot(df["Frame"], df[y_col].values, label="Y", linewidth=0.7, alpha=0.8)
        valid = True
    if z_col in df.columns:
        ax.plot(df["Frame"], df[z_col].values, label="Z", linewidth=0.7, alpha=0.8)
        valid = True
    ax.set_xlabel("Frame")
    ax.set_ylabel("Position (m)")
    ax.grid(True, alpha=0.3)
    if valid:
        ax.legend(loc="upper right", fontsize=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot preprocessed ZED hand-tracking CSVs: one window per hand, subplots per camera."
    )
    parser.add_argument(
        "--take",
        default=None,
        help="Take name (default: from config.json)",
    )
    args = parser.parse_args()

    take = args.take or config.take_name
    data_dir = "data/dataframes"
    pattern = os.path.join(data_dir, f"{take}_cam*_preprocessed.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        print(f"No preprocessed CSV files found matching {pattern}")
        sys.exit(1)

    cam_map = {}
    for path in csv_files:
        basename = os.path.splitext(os.path.basename(path))[0]
        m = re.search(r"_cam(\d+)_preprocessed$", basename)
        if m:
            idx = int(m.group(1))
            cam_map[idx] = path

    if not cam_map:
        print("Could not parse camera indices from filenames")
        sys.exit(1)

    cam_indices = sorted(cam_map.keys())
    n_cams = len(cam_indices)
    take_name = take

    dfs = {idx: pd.read_csv(cam_map[idx]) for idx in cam_indices}

    for hand in ("left", "right"):
        fig, axes = plt.subplots(n_cams, len(SIGNIFICANT_JOINTS),
                                 figsize=(14, 4 * n_cams), squeeze=False)
        fig.suptitle(f"{take_name} — {hand.capitalize()} Hand — Preprocessed",
                     fontsize=14, fontweight="bold")

        for row, cam_idx in enumerate(cam_indices):
            df = dfs[cam_idx]
            for col_idx, joint_idx in enumerate(SIGNIFICANT_JOINTS):
                ax = axes[row, col_idx]
                if row == 0:
                    ax.set_title(JOINT_LABELS[col_idx])
                _plot_joint(ax, df, hand, joint_idx)
                if col_idx == 0:
                    ax.set_ylabel(f"cam{cam_idx}")

        plt.tight_layout()

    plt.show()
