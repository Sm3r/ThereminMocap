import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import config


TARGETS = ["pitch", "volume"]
HAND_MAP = {"pitch": "right", "volume": "left"}


def _build_columns(raw_path):
    with open(raw_path, "r") as f:
        reader = csv.reader(f)
        header_rows = [next(reader) for _ in range(8)]

    raw_types = header_rows[2]
    raw_names = header_rows[3]
    raw_subtype = header_rows[6]
    raw_headers = header_rows[7]

    all_names = []
    keep_mask = []

    for i in range(1, len(raw_headers)):
        t = raw_types[i].strip() if i < len(raw_types) else ""
        n = raw_names[i].strip() if i < len(raw_names) else ""
        sub = raw_subtype[i].strip() if i < len(raw_subtype) else ""
        h = raw_headers[i].strip() if i < len(raw_headers) else ""

        if "Unlabeled" in n:
            all_names.append(None)
            keep_mask.append(False)
        elif sub == "Position" and h in ("X", "Y", "Z"):
            col_name = f"{t}_{n}_{h}".replace(" ", "_").replace(":", "_")
            all_names.append(col_name)
            keep_mask.append(True)
        else:
            all_names.append(None)
            keep_mask.append(False)

    return all_names, keep_mask


def _hand_prefixes(target):
    hand = HAND_MAP[target]
    bone_prefix = f"Bone_{hand}_{hand}"
    bone_marker_prefix = f"Bone_Marker_{hand}_Marker"
    marker_prefix = f"Marker_{hand}_Marker"
    return bone_prefix, bone_marker_prefix, marker_prefix


def plot_markers(target):
    hand = HAND_MAP[target]
    take_name = config.get_take_name(target)
    raw_path = f"data/features/mocap/OPTITRACK_{take_name}_raw.csv"

    if not os.path.exists(raw_path):
        print(f"  Skipping {target}: {raw_path} not found")
        return

    all_names, keep_mask = _build_columns(raw_path)
    num_meta_cols = len(all_names)

    df = pd.read_csv(raw_path, skiprows=8, header=None,
                     on_bad_lines='skip')

    if df.shape[1] < num_meta_cols:
        num_meta_cols = df.shape[1]
        all_names = all_names[:num_meta_cols]
        keep_mask = keep_mask[:num_meta_cols]

    col_names = []
    drop_idx = []
    for i in range(num_meta_cols):
        if keep_mask[i]:
            col_names.append(all_names[i])
        else:
            drop_idx.append(i)

    extra_cols = df.shape[1] - num_meta_cols
    if extra_cols > 0:
        drop_idx.extend(range(num_meta_cols, df.shape[1]))

    df = df.drop(columns=drop_idx, errors='ignore')
    df.columns = col_names

    bone_prefix, bone_marker_prefix, marker_prefix = _hand_prefixes(target)

    marker_groups = []
    marker_labels = []
    for col_name in col_names:
        if col_name.startswith(bone_prefix) or col_name.startswith(bone_marker_prefix) or col_name.startswith(marker_prefix):
            base = col_name.rsplit("_", 1)[0]
            if base not in marker_groups:
                marker_groups.append(base)
                short = base.replace(bone_prefix, "Bone").replace(bone_marker_prefix, "BoneM").replace(marker_prefix, "M")
                marker_labels.append(short)

    if not marker_groups:
        print(f"  Skipping {target}: no hand markers found")
        return

    num_markers = len(marker_groups)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f'{take_name} — {hand} hand (raw, no Unlabeled)',
                 fontsize=16, fontweight='bold')

    ax_x = plt.subplot(3, 1, 1)
    ax_y = plt.subplot(3, 1, 2)
    ax_z = plt.subplot(3, 1, 3)

    colors = plt.cm.Set1(np.linspace(0, 0.9, num_markers))

    for mi, (base, label) in enumerate(zip(marker_groups, marker_labels)):
        color = colors[mi]
        for row, (ax, axis) in enumerate([(ax_x, '_X'), (ax_y, '_Y'), (ax_z, '_Z')]):
            col = f"{base}{axis}"
            if col in df.columns:
                vals = df[col].values
                ax.plot(vals, linewidth=0.8, alpha=0.7, color=color,
                        label=label if row == 0 else "")

    ax_x.set_ylabel('X Position')
    ax_x.grid(True, alpha=0.3)
    ax_x.set_title('X Axis')
    ax_x.legend(loc='upper right', fontsize=8)

    ax_y.set_ylabel('Y Position')
    ax_y.grid(True, alpha=0.3)
    ax_y.set_title('Y Axis')

    ax_z.set_ylabel('Z Position')
    ax_z.set_xlabel('Frame Number')
    ax_z.grid(True, alpha=0.3)
    ax_z.set_title('Z Axis')

    fig.tight_layout()
    plt.show(block=False)
    return fig


if __name__ == "__main__":
    figs = []
    for target in TARGETS:
        fig = plot_markers(target)
        if fig is not None:
            figs.append(fig)
    if figs:
        plt.show()
