import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import config


SIGNIFICANT_JOINTS = [0, 5, 17]
JOINT_LABELS = ["Wrist (0)", "Index MCP (5)", "Pinky MCP (17)"]
COLORS = {"X": "tab:red", "Y": "tab:green", "Z": "tab:blue"}
HAND_MAP = {"pitch": "right", "volume": "left"}
TARGETS = ["pitch", "volume"]


def plot_target(target):
    hand = HAND_MAP[target]
    take_name = config.get_take_name(target)
    data_dir = "data/features"

    csv_path = os.path.join(data_dir, f"{take_name}_cam1.csv")
    npy_path = os.path.join(data_dir, f"{take_name}_hand.npy")

    if not os.path.exists(csv_path):
        print(f"  Skipping {target}: {csv_path} not found")
        return
    if not os.path.exists(npy_path):
        print(f"  Skipping {target}: {npy_path} not found")
        return

    df_raw = pd.read_csv(csv_path)
    hand_cols = sorted(c for c in df_raw.columns
                       if c.startswith(f"{hand}_") and "_detected" not in c)
    if not hand_cols:
        print(f"  Skipping {target}: no {hand} hand columns in CSV")
        return

    raw_data = df_raw[hand_cols].values
    proc_data = np.load(npy_path)

    n = min(len(raw_data), len(proc_data))
    raw_data = raw_data[:n]
    proc_data = proc_data[:n]

    fig, axes = plt.subplots(2, len(SIGNIFICANT_JOINTS),
                             figsize=(14, 6), squeeze=False)
    fig.suptitle(f"{take_name} — {hand} hand: Raw (CSV) vs Processed (NPY)",
                 fontsize=14, fontweight="bold")

    for stage_idx, (data, label) in enumerate([(raw_data, "raw"), (proc_data, "proc")]):
        for col_idx, joint_idx in enumerate(SIGNIFICANT_JOINTS):
            ax = axes[stage_idx, col_idx]

            if stage_idx == 0:
                ax.set_title(JOINT_LABELS[col_idx])

            ax.set_ylabel(label, fontsize=8)
            ax.set_xlabel("Frame")
            ax.grid(True, alpha=0.3)

            for ax_idx, axis_label in enumerate(["X", "Y", "Z"]):
                col_in_npy = joint_idx * 3 + ax_idx
                vals = data[:, col_in_npy]
                ax.plot(range(len(vals)), vals,
                        color=COLORS[axis_label],
                        linewidth=0.8, alpha=0.85,
                        label=axis_label if col_idx == 0 else "")

        axes[stage_idx, 0].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    plt.show(block=False)


if __name__ == "__main__":
    for target in TARGETS:
        plot_target(target)

    if any(os.path.exists(f"data/features/{config.get_take_name(t)}_cam1.csv") and
           os.path.exists(f"data/features/{config.get_take_name(t)}_hand.npy")
           for t in TARGETS):
        plt.show()
