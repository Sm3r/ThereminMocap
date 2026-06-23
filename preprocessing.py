import numpy as np
import pandas as pd

from config import config

def fix_hand_labels(df: pd.DataFrame, target_hand: str = "right",
                    alpha: float = 0.3, threshold: float = 0.3) -> pd.DataFrame:
    df = df.copy()
    other_hand = "left" if target_hand == "right" else "right"

    target_x = df[f"{target_hand}_00_X"].values
    target_y = df[f"{target_hand}_00_Y"].values
    target_z = df[f"{target_hand}_00_Z"].values
    target_detected = df[f"{target_hand}_2d_detected"].values

    other_x = df[f"{other_hand}_00_X"].values
    other_y = df[f"{other_hand}_00_Y"].values
    other_z = df[f"{other_hand}_00_Z"].values
    other_detected = df[f"{other_hand}_2d_detected"].values

    ema = None
    swap_mask = np.zeros(len(df), dtype=bool)

    for i in range(len(df)):
        t_det = target_detected[i]
        o_det = other_detected[i]

        if t_det and not np.isnan(target_x[i]):
            pos = np.array([target_x[i], target_y[i], target_z[i]])
            if np.isnan(pos).any():
                continue
            if ema is None:
                ema = pos.copy()
            else:
                ema = alpha * pos + (1 - alpha) * ema

        elif o_det and ema is not None and not np.isnan(other_x[i]):
            other_pos = np.array([other_x[i], other_y[i], other_z[i]])
            if np.isnan(other_pos).any():
                continue
            if np.linalg.norm(other_pos - ema) < threshold:
                swap_mask[i] = True
                ema = alpha * other_pos + (1 - alpha) * ema

    n_swaps = swap_mask.sum()
    if n_swaps > 0:
        left_cols = sorted(c for c in df.columns if c.startswith("left_"))
        right_cols = sorted(c for c in df.columns if c.startswith("right_"))
        left_vals = df.loc[swap_mask, left_cols].copy().values
        right_vals = df.loc[swap_mask, right_cols].copy().values
        df.loc[swap_mask, left_cols] = right_vals
        df.loc[swap_mask, right_cols] = left_vals
        print(f"\n    Swapped {n_swaps}/{len(df)} frames ({n_swaps / len(df) * 100:.1f}%)")

    return df


def drop_minority_hand(df: pd.DataFrame, min_ratio: float = 3.0) -> pd.DataFrame:
    """Drop frames where the minority hand is detected.

    Each frame records `left_2d_detected` / `right_2d_detected`.
    The hand detected in the majority of frames is the dominant one;
    frames where the dominant hand was **not** detected are dropped.

    Parameters
    ----------
    df : pd.DataFrame — input data
    min_ratio : float — minimum majority/minority ratio to apply dropping

    Returns
    -------
    pd.DataFrame — filtered DataFrame
    """
    has_left = "left_2d_detected" in df.columns
    has_right = "right_2d_detected" in df.columns
    if not has_left and not has_right:
        return df
    left_count = df["left_2d_detected"].sum() if has_left else 0
    right_count = df["right_2d_detected"].sum() if has_right else 0

    if left_count == 0 and right_count == 0:
        print(f"    No hand detections found, skipping")
        return df

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
        return df

    keep = df[f"{majority_hand}_2d_detected"].astype(bool)
    n_before = len(df)
    df = df[keep].reset_index(drop=True)
    n_dropped = n_before - len(df)
    print(f"    Dropped {n_dropped}/{n_before} frames")
    return df


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

def remove_spikes(df: pd.DataFrame, window: int = 128, threshold: float = 5.0, iterations: int = 2) -> pd.DataFrame:
    df = df.copy()
    hand_cols = []

    for hand in ("left", "right"):
        for i in range(21):
            for axis in ("X", "Y", "Z"):
                col = f"{hand}_{i:02d}_{axis}"
                if col in df.columns:
                    hand_cols.append(col)

    for _ in range(iterations):
        if _ == 0:
            window = window
            threshold = 5.0
        else:
            window = window // 2
            threshold = 5.0
        mask = _create_spike_mask(df, hand_cols, window, threshold)
        for col in hand_cols:
            n_valid = df[col].notna().sum()
            if n_valid < 0.05 * len(df):
                continue
            df.loc[mask[col], col] = np.nan

    return df
