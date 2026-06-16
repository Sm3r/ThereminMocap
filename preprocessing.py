import glob
import os
import sys

import numpy as np
import pandas as pd

from config import config


def _create_spike_mask(df: pd.DataFrame, columns: list[str],
                       window: int, threshold: float) -> pd.DataFrame:
    half = window // 2
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col in columns:
        if col not in df.columns:
            continue
        roll_med = df[col].rolling(window=window, center=True, min_periods=1).median()
        roll_mad = (df[col] - roll_med).abs().rolling(
            window=window, center=True, min_periods=1
        ).median()
        dev = (df[col] - roll_med).abs()
        mask[col] = (roll_mad > 0) & (dev / roll_mad > threshold)
    return mask


def fix_hand_labels(csv_path: str, window: int = 30, min_votes: int = 10, swap_ratio: float = 2.0) -> str:
    print(f"  Correcting hand swaps in {os.path.basename(csv_path)} ...")
    df = pd.read_csv(csv_path)

    left_x = df["left_00_X"].values
    left_y = df["left_00_Y"].values
    right_x = df["right_00_X"].values
    right_y = df["right_00_Y"].values

    valid = ~(np.isnan(left_x) | np.isnan(left_y) | np.isnan(right_x) | np.isnan(right_y))

    swap_mask = np.zeros(len(df), dtype=bool)

    for i in range(len(df)):
        if not valid[i]:
            continue

        left_agree = 0
        left_disagree = 0
        right_agree = 0
        right_disagree = 0

        start = max(0, i - window)
        end = min(len(df), i + window + 1)

        for j in range(start, end):
            if i == j or not valid[j]:
                continue

            d_ll = np.hypot(left_x[i] - left_x[j], left_y[i] - left_y[j])
            d_lr = np.hypot(left_x[i] - right_x[j], left_y[i] - right_y[j])

            if d_ll <= d_lr:
                left_agree += 1
            else:
                left_disagree += 1

            d_rl = np.hypot(right_x[i] - left_x[j], right_y[i] - left_y[j])
            d_rr = np.hypot(right_x[i] - right_x[j], right_y[i] - right_y[j])

            if d_rr <= d_rl:
                right_agree += 1
            else:
                right_disagree += 1

        total_left = left_agree + left_disagree
        total_right = right_agree + right_disagree

        left_swap = total_left >= min_votes and left_disagree > left_agree * swap_ratio
        right_swap = total_right >= min_votes and right_disagree > right_agree * swap_ratio

        if left_swap and right_swap:
            swap_mask[i] = True

    n_swaps = swap_mask.sum()

    if n_swaps > 0:
        left_cols = sorted(c for c in df.columns if c.startswith("left_"))
        right_cols = sorted(c for c in df.columns if c.startswith("right_"))

        left_vals = df.loc[swap_mask, left_cols].copy().values
        right_vals = df.loc[swap_mask, right_cols].copy().values

        df.loc[swap_mask, left_cols] = right_vals
        df.loc[swap_mask, right_cols] = left_vals

        print(f"    Swapped {n_swaps}/{len(df)} frames ({n_swaps / len(df) * 100:.1f}%)")

    df.to_csv(csv_path, index=False)
    return csv_path


def drop_minority_hand(csv_path: str, min_ratio: float = 3.0) -> str:
    """Drop frames where the minority hand is detected.

    Each frame records `left_2d_detected` / `right_2d_detected`.
    The hand detected in the majority of frames is the dominant one;
    frames where the dominant hand was **not** detected are dropped.

    Parameters
    ----------
    csv_path : str — path to CSV (modified in-place)
    min_ratio : float — minimum majority/minority ratio to apply dropping

    Returns
    -------
    str — path to the modified CSV
    """
    df = pd.read_csv(csv_path)

    has_left = "left_2d_detected" in df.columns
    has_right = "right_2d_detected" in df.columns
    if not has_left and not has_right:
        return csv_path
    left_count = df["left_2d_detected"].sum() if has_left else 0
    right_count = df["right_2d_detected"].sum() if has_right else 0

    if left_count == 0 and right_count == 0:
        print(f"    No hand detections found, skipping")
        return csv_path

    if left_count > right_count:
        majority_hand = "left"
        maj_count = left_count
        ratio = left_count / right_count if right_count > 0 else float("inf")
    else:
        majority_hand = "right"
        maj_count = right_count
        ratio = right_count / left_count if left_count > 0 else float("inf")

    print(f"    Majority hand: {majority_hand} ({maj_count}/{len(df)} frames, ratio={ratio:.1f})")

    if ratio < min_ratio:
        print(f"    Ratio {ratio:.1f} < {min_ratio}, skipping drop")
        return csv_path

    keep = df[f"{majority_hand}_2d_detected"].astype(bool)
    n_before = len(df)
    df = df[keep].reset_index(drop=True)
    n_dropped = n_before - len(df)
    print(f"    Dropped {n_dropped}/{n_before} frames")
    df.to_csv(csv_path, index=False)
    return csv_path


def preprocess_csv(csv_path: str, window: int = 128, threshold: float = 8.0) -> str:
    print(f"  Cleaning {os.path.basename(csv_path)} ...")
    df = pd.read_csv(csv_path)
    hand_cols = []

    for hand in ("left", "right"):
        for i in range(21):
            for axis in ("X", "Y", "Z"):
                col = f"{hand}_{i:02d}_{axis}"
                if col in df.columns:
                    hand_cols.append(col)

    for _ in range(2):
        if range == 0:
            window = 128
            threshold = 8.0
        else:
            window = 64
            threshold = 8.0
        mask = _create_spike_mask(df, hand_cols, window, threshold)
        for col in hand_cols:
            n_valid = df[col].notna().sum()
            if n_valid < 0.05 * len(df):
                continue
            df.loc[mask[col], col] = np.nan

    base, ext = os.path.splitext(csv_path)
    out_path = f"{base}_preprocessed{ext}"
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    take_name = config.take_name
    pattern = os.path.join("data", "dataframes", f"{take_name}_cam*.csv")
    csv_files = sorted(
        f for f in glob.glob(pattern)
        if not os.path.splitext(f)[0].endswith("_preprocessed")
    )

    if not csv_files:
        print(f"No CSV files found matching {pattern}")
        sys.exit(1)

    print(f"Preprocessing {len(csv_files)} CSV(s) for take '{take_name}' ...\n")
    for csv_path in csv_files:
        out_path = preprocess_csv(csv_path)
        fix_hand_labels(out_path)
