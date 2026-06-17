import os
import numpy as np
import pandas as pd
from hand_tracking_ZED6D.capture import capture_to_csv
from preprocessing import fix_hand_labels, remove_spikes
from config import config

# ==========================
# CONFIG
# ==========================
fix_hand_label = True
remove_spike = True
show_windows = False

_HAND_MAP = {"pitch": "right", "volume": "left"}

os.makedirs("data/features", exist_ok=True)

for target in ("pitch", "volume"):
    take_name = config.get_take_name(target)
    target_hand = _HAND_MAP[target]

    svo_path = f"data/recordings/{take_name}_cam1.svo2"
    if not os.path.exists(svo_path):
        print(f"  Skipping {take_name}: {svo_path} not found")
        continue

    csv_path = f"data/features/{take_name}_cam1.csv"

    if not os.path.exists(csv_path):
        print(f"  Processing {take_name} ...")
        capture_to_csv(
            filename=svo_path,
            output_csv=csv_path,
            show_windows=show_windows,
            fps=config.rates.zed_fps,
            use_triangulation=(config.depth_mode == "triangulation"),
        )

    df = pd.read_csv(csv_path)

    print(f"\n  === Detection summary: {take_name} ({target_hand} hand) ===")
    if fix_hand_label:
        df = fix_hand_labels(df, target_hand=target_hand)
    if remove_spike:
        df = remove_spikes(df)

    # Detection summary for target hand
    total = len(df)
    det_col = f"{target_hand}_2d_detected"
    if det_col in df.columns:
        detected = df[det_col].sum()
        pct_det = f"{detected / total * 100:5.1f}%" if total else "N/A"
    else:
        detected, pct_det = total, "100.0%"

    hand_cols = sorted(c for c in df.columns
                       if c.startswith(f"{target_hand}_") and "_detected" not in c)
    depth_valid = df[hand_cols].notna().any(axis=1).sum() if hand_cols else 0
    pct_depth = f"{depth_valid / total * 100:5.1f}%" if total else "N/A"

    print(f"    Total frames:        {total}")
    print(f"    MediaPipe detected:   {detected:>6} / {total} ({pct_det})")
    print(f"    3D depth valid:       {depth_valid:>6} / {total} ({pct_depth})")

    if hand_cols:
        np.save(f"data/features/{take_name}_hand.npy", df[hand_cols].values)
        print(f"    Saved {target_hand} hand: {df[hand_cols].shape}")
