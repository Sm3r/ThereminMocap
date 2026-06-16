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
COLORS = {"X": "tab:red", "Y": "tab:green", "Z": "tab:blue"}


def _dominant_hand(df):
    left = df.get("left_2d_detected", pd.Series([0])).sum()
    right = df.get("right_2d_detected", pd.Series([0])).sum()
    if left > right:
        return "left"
    return "right"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot raw vs preprocessed ZED hand-tracking CSVs: "
                    "one row per processing stage per camera."
    )
    parser.add_argument("--take", default=None, help="Take name (default: from config.json)")
    args = parser.parse_args()

    take = args.take or config.take_name
    data_dir = "data/dataframes"

    raw_pattern = os.path.join(data_dir, f"{take}_cam*.csv")
    raw_files = sorted(glob.glob(raw_pattern))

    cam_raw = {}
    cam_prep = {}
    for path in raw_files:
        base = os.path.splitext(os.path.basename(path))[0]
        if base.endswith("_world") or base.endswith("_preprocessed"):
            continue
        m = re.search(r"_cam(\d+)$", base)
        if not m:
            continue
        idx = int(m.group(1))
        raw_path = path
        prep_path = os.path.join(data_dir, f"{base}_preprocessed.csv")
        if not os.path.exists(prep_path):
            print(f"  Skipping cam{idx} — no preprocessed file found")
            continue
        cam_raw[idx] = raw_path
        cam_prep[idx] = prep_path

    if not cam_raw:
        print(f"No camera CSV pairs found matching {raw_pattern}")
        sys.exit(1)

    cam_indices = sorted(cam_raw.keys())
    n_cams = len(cam_indices)
    n_rows = n_cams * 2
    n_cols = len(SIGNIFICANT_JOINTS)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(14, 3 * n_rows), squeeze=False)
    cam_hand_parts = []
    for idx in cam_indices:
        df = pd.read_csv(cam_raw[idx])
        hand = _dominant_hand(df)
        cam_hand_parts.append(f"cam{idx} {hand}")
    fig.suptitle(f"Raw vs Preprocessed: {take} — {' / '.join(cam_hand_parts)} hand ",
                 fontsize=14, fontweight="bold")

    for cam_row, cam_idx in enumerate(cam_indices):
        df_raw = pd.read_csv(cam_raw[cam_idx])
        df_prep = pd.read_csv(cam_prep[cam_idx])

        hand = _dominant_hand(df_raw)

        # Filter raw to same rows as preprocessed (dominant-hand frames)
        det_col = f"{hand}_2d_detected"
        if det_col in df_raw.columns:
            df_raw = df_raw[df_raw[det_col].astype(bool)].reset_index(drop=True)

        n = min(len(df_raw), len(df_prep))
        df_raw = df_raw.iloc[:n]
        df_prep = df_prep.iloc[:n]

        for stage_idx, (df, stage) in enumerate([(df_raw, "raw"), (df_prep, "preproc")]):
            row = cam_row * 2 + stage_idx

            for col_idx, joint_idx in enumerate(SIGNIFICANT_JOINTS):
                ax = axes[row, col_idx]

                if cam_row == 0:
                    ax.set_title(JOINT_LABELS[col_idx])

                ax.set_ylabel(stage, fontsize=8)

                for axis_label in ["X", "Y", "Z"]:
                    col = f"{hand}_{joint_idx:02d}_{axis_label}"
                    if col in df.columns:
                        ax.plot(df.index, df[col].values,
                                color=COLORS[axis_label],
                                linewidth=0.8, alpha=0.85,
                                label=axis_label if col_idx == 0 else "")

                ax.set_xlabel("Frame")
                ax.grid(True, alpha=0.3)

            axes[row, 0].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()
