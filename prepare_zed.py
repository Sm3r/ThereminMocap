import os
import numpy as np
import pandas as pd
from hand_tracking_ZED6D.capture import capture_to_csv
from preprocessing import fix_hand_labels, drop_minority_hand, remove_spikes
from config import config

# ==========================
# CONFIG
# ==========================
fix_hand_label = True
drop_min_hand = False
remove_spikes = True
show_windows = False

os.makedirs("data/features", exist_ok=True)

for target in ("pitch", "volume"):
    take_name = config.get_take_name(target)

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

    # Apply preprocessing
    if fix_hand_label:
        df = fix_hand_labels(df)
    if drop_min_hand:
        df = drop_minority_hand(df)
    if remove_spikes:
        df = remove_spikes(df)

    df.to_csv(csv_path, index=False)

    # Detection summary
    total = len(df)
    has_2d = 'left_2d_detected' in df.columns and 'right_2d_detected' in df.columns

    landmark_cols = sorted(c for c in df.columns
                           if (c.startswith("left_") or c.startswith("right_"))
                           and "_detected" not in c)
    depth_valid = df[landmark_cols].notna().any(axis=1).sum() if landmark_cols else 0

    basename = os.path.basename(csv_path)
    pct_depth = f"{depth_valid / total * 100:5.1f}%" if total else "N/A"

    print(f"\n  === Detection summary: {basename} ===")
    print(f"    Total frames:            {total}")
    if has_2d:
        mp_detected = df[['left_2d_detected', 'right_2d_detected']].any(axis=1).sum()
        pct_mp = f"{mp_detected / total * 100:5.1f}%" if total else "N/A"
        print(f"    MediaPipe 2D detected:   {mp_detected:>6} / {total} ({pct_mp})")
    print(f"    Wrist depth valid:       {depth_valid:>6} / {total} ({pct_depth})")

    # Save hands features
    left_cols = sorted(c for c in df.columns
                       if c.startswith("left_") and "_detected" not in c)
    right_cols = sorted(c for c in df.columns
                        if c.startswith("right_") and "_detected" not in c)

    if left_cols:
        np.save(f"data/features/{take_name}_left_hand.npy", df[left_cols].values)
        print(f"    Saved left hand:  {df[left_cols].shape}")
    if right_cols:
        np.save(f"data/features/{take_name}_right_hand.npy", df[right_cols].values)
        print(f"    Saved right hand: {df[right_cols].shape}")
