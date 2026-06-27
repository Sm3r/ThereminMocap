import argparse
import csv
import glob
import math
import os
import random
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from network import HandNet


DEFAULT_FEATURE_DIR = "data/features"
DEFAULT_OUTPUT_DIR = "runs_mocap_multi_cv_dual"

HAND_FPS = 360
TARGET_FPS = 60.0
DEFAULT_MAX_NILS_TO_FILL = -1

AXES = ("X", "Y", "Z")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def assert_not_lfs_pointer(path: str) -> None:
    with open(path, "rb") as f:
        head = f.read(128)

    if head.startswith(b"version https://git-lfs.github.com/spec"):
        raise RuntimeError(
            "This file is a Git LFS pointer, not a real data file:\n"
            f"  {path}\n"
            "Download the real LFS object first, then rerun this script."
        )


def coerce_array_to_float32(arr) -> np.ndarray:
    if isinstance(arr, np.ndarray) and arr.shape == ():
        arr = arr.item()

    arr = np.asarray(arr)

    if arr.dtype == object:
        arr = np.where(arr == "nil", np.nan, arr)
        arr = np.where(arr == "None", np.nan, arr)
        arr = np.where(arr == "", np.nan, arr)

    return arr.astype(np.float32)


def load_npy_array(path: str) -> np.ndarray:
    assert_not_lfs_pointer(path)
    try:
        arr = np.load(path, allow_pickle=True)
    except Exception as exc:
        raise RuntimeError(f"Could not load NumPy file: {path}\nOriginal error: {exc}") from exc
    return coerce_array_to_float32(arr)


def load_csv_array(path: str, csv_target_column: int) -> np.ndarray:
    arr = np.genfromtxt(path, delimiter=",", dtype=np.float32)

    if arr.ndim == 2:
        all_nan_rows = np.all(~np.isfinite(arr), axis=1)
        arr = arr[~all_nan_rows]
    elif arr.ndim == 1:
        arr = arr[np.isfinite(arr)]

    if arr.ndim == 2 and arr.shape[1] > 1:
        arr = arr[:, csv_target_column]

    return coerce_array_to_float32(arr)


def load_target_array(path: str, csv_target_column: int) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return load_npy_array(path)
    if ext == ".csv":
        return load_csv_array(path, csv_target_column=csv_target_column)
    raise RuntimeError(f"Unsupported file extension for {path}. Expected .npy or .csv.")


def safe_float(value: str) -> float:
    value = value.strip()
    if value == "" or value.lower() in {"nil", "none", "nan"}:
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def load_mocap_csv_matrix(path: str) -> Tuple[List[str], np.ndarray]:
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise RuntimeError(f"Empty MOCAP CSV file: {path}") from exc

        header = [col.strip() for col in header]
        rows = []

        for row in reader:
            if not row:
                continue

            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            elif len(row) > len(header):
                row = row[:len(header)]

            rows.append([safe_float(value) for value in row])

    if not rows:
        raise RuntimeError(f"MOCAP CSV file has no data rows: {path}")

    data = np.asarray(rows, dtype=np.float32)
    return header, data


# -----------------------------------------------------------------------------
# MOCAP column handling
# -----------------------------------------------------------------------------

def category_from_stimulus(stimulus: str) -> Optional[str]:
    name = stimulus.lower()
    is_pitch = "pitch" in name
    is_volume = "volume" in name

    if is_pitch and is_volume:
        raise RuntimeError(f"Stimulus name matches both pitch and volume: {stimulus}")
    if is_pitch:
        return "pitch"
    if is_volume:
        return "volume"
    return None


def strip_known_mocap_suffixes(stem: str) -> List[str]:
    candidates = [stem]
    suffixes = ["_cleaned", "_mocap", "_features"]

    changed = True
    current = stem
    while changed:
        changed = False
        for suffix in suffixes:
            if current.endswith(suffix):
                current = current[:-len(suffix)]
                if current and current not in candidates:
                    candidates.append(current)
                changed = True

    return candidates


def marker_has_xyz(header_set: set, marker: str) -> bool:
    return all(f"{marker}_{axis}" in header_set for axis in AXES)


def available_marker_ids(header: Sequence[str], prefix: str) -> List[str]:
    header_set = set(header)
    marker_ids = []

    pattern = re.compile(rf"^{re.escape(prefix)}_(\d{{3}})_X$")
    for col in header:
        match = pattern.match(col)
        if not match:
            continue
        marker_id = match.group(1)
        marker_name = f"{prefix}_{marker_id}"
        if marker_has_xyz(header_set, marker_name):
            marker_ids.append(marker_id)

    return sorted(set(marker_ids))


def candidate_marker_prefixes(header: Sequence[str]) -> List[str]:
    header_set = set(header)
    prefixes = set()
    pattern = re.compile(r"^(.+)_(\d{3})_X$")

    for col in header:
        match = pattern.match(col)
        if not match:
            continue
        prefix = match.group(1)
        marker_id = match.group(2)
        marker_name = f"{prefix}_{marker_id}"
        if marker_has_xyz(header_set, marker_name):
            prefixes.add(prefix)

    return sorted(prefixes)


def resolve_hand_markers(
    header: Sequence[str],
    category: str,
    hand_prefixes: Sequence[str],
    pitch_hand_max_markers: int,
    volume_hand_max_markers: int,
) -> Tuple[str, List[str]]:
    for prefix in hand_prefixes:
        marker_ids = available_marker_ids(header, prefix)
        if marker_ids:
            max_markers = pitch_hand_max_markers if category == "pitch" else volume_hand_max_markers
            if max_markers > 0:
                marker_ids = marker_ids[:max_markers]
            markers = [f"{prefix}_{marker_id}" for marker_id in marker_ids]
            return prefix, markers

    raise RuntimeError(
        "Could not find hand marker columns. Tried prefixes: "
        f"{list(hand_prefixes)}"
    )


def resolve_antenna_prefix(
    header: Sequence[str],
    category: str,
    pitch_antenna_prefix: Optional[str],
    volume_antenna_prefix: Optional[str],
) -> str:
    header_set = set(header)

    override = pitch_antenna_prefix if category == "pitch" else volume_antenna_prefix
    if override:
        required = [f"{override}_{idx:03d}" for idx in range(1, 4)]
        if all(marker_has_xyz(header_set, marker) for marker in required):
            return override
        raise RuntimeError(
            f"Requested {category} antenna prefix {override!r}, but the CSV does not "
            "contain markers 001, 002, and 003 with X/Y/Z columns."
        )

    prefixes = candidate_marker_prefixes(header)

    excluded_tokens = ["right", "left", "hand", "webcam", "camera", "zed"]
    valid_prefixes = []
    for prefix in prefixes:
        prefix_l = prefix.lower()
        if any(token in prefix_l for token in excluded_tokens):
            continue
        required = [f"{prefix}_{idx:03d}" for idx in range(1, 4)]
        if all(marker_has_xyz(header_set, marker) for marker in required):
            valid_prefixes.append(prefix)

    exact = [prefix for prefix in valid_prefixes if prefix.lower() == category]
    if exact:
        return exact[0]

    if category == "volume":
        vol_exact = [prefix for prefix in valid_prefixes if prefix.lower() == "vol"]
        if vol_exact:
            return vol_exact[0]

    contains_category = [prefix for prefix in valid_prefixes if category in prefix.lower()]
    if contains_category:
        return contains_category[0]

    raise RuntimeError(
        f"Could not auto-detect {category} antenna markers. "
        "Use --pitch-antenna-prefix or --volume-antenna-prefix."
    )


def extract_marker_xyz(
    data: np.ndarray,
    header_to_index: Dict[str, int],
    marker: str,
) -> np.ndarray:
    cols = [header_to_index[f"{marker}_{axis}"] for axis in AXES]
    return data[:, cols].astype(np.float32)


def extract_frame_column(data: np.ndarray, header_to_index: Dict[str, int]) -> np.ndarray:
    if "Frame" in header_to_index:
        frames = data[:, header_to_index["Frame"]]
        valid = np.isfinite(frames)
        out = np.arange(len(data), dtype=np.int64)
        out[valid] = np.rint(frames[valid]).astype(np.int64)
        return out

    return np.arange(len(data), dtype=np.int64)


def compute_antenna_center_and_scale(
    antenna_xyz: np.ndarray,
    min_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    center = np.nanmean(antenna_xyz, axis=1).astype(np.float32)

    d01 = np.linalg.norm(antenna_xyz[:, 0, :] - antenna_xyz[:, 1, :], axis=1)
    d02 = np.linalg.norm(antenna_xyz[:, 0, :] - antenna_xyz[:, 2, :], axis=1)
    d12 = np.linalg.norm(antenna_xyz[:, 1, :] - antenna_xyz[:, 2, :], axis=1)

    pairwise = np.stack([d01, d02, d12], axis=1)
    scale = np.nanmax(pairwise, axis=1).astype(np.float32)

    valid_scale = np.isfinite(scale) & (scale > min_scale)
    if np.any(valid_scale):
        fallback_scale = float(np.nanmedian(scale[valid_scale]))
    else:
        raise RuntimeError("Could not compute a valid antenna scale from the three antenna markers.")

    scale[~valid_scale] = fallback_scale
    scale[scale < min_scale] = fallback_scale

    return center, scale.astype(np.float32)


def load_mocap_features(
    path: str,
    stimulus: str,
    args,
) -> Dict:
    category = category_from_stimulus(stimulus)
    if category is None:
        raise RuntimeError(f"Could not infer category from stimulus name: {stimulus}")

    header, data = load_mocap_csv_matrix(path)
    header_to_index = {name: idx for idx, name in enumerate(header)}

    frames = extract_frame_column(data, header_to_index)

    hand_prefix, hand_markers = resolve_hand_markers(
        header=header,
        category=category,
        hand_prefixes=args.hand_prefixes,
        pitch_hand_max_markers=args.pitch_hand_max_markers,
        volume_hand_max_markers=args.volume_hand_max_markers,
    )

    antenna_prefix = resolve_antenna_prefix(
        header=header,
        category=category,
        pitch_antenna_prefix=args.pitch_antenna_prefix,
        volume_antenna_prefix=args.volume_antenna_prefix,
    )

    antenna_markers = [f"{antenna_prefix}_{idx:03d}" for idx in range(1, 4)]

    hand_xyz = np.stack(
        [extract_marker_xyz(data, header_to_index, marker) for marker in hand_markers],
        axis=1,
    )
    antenna_xyz = np.stack(
        [extract_marker_xyz(data, header_to_index, marker) for marker in antenna_markers],
        axis=1,
    )

    antenna_center, antenna_scale = compute_antenna_center_and_scale(
        antenna_xyz=antenna_xyz,
        min_scale=args.min_antenna_scale,
    )

    relative = hand_xyz - antenna_center[:, None, :]
    relative = relative / antenna_scale[:, None, None]
    x_relative = relative.reshape(relative.shape[0], -1).astype(np.float32)

    print()
    print("MOCAP feature extraction")
    print(f"stimulus:        {stimulus}")
    print(f"category:        {category}")
    print(f"mocap path:      {path}")
    print(f"frames:          {len(frames)}")
    print(f"hand prefix:     {hand_prefix}")
    print(f"hand markers:    {', '.join(hand_markers)}")
    print(f"antenna prefix:  {antenna_prefix}")
    print(f"antenna markers: {', '.join(antenna_markers)}")
    print(f"feature shape:   {x_relative.shape}")
    print(f"antenna scale median: {float(np.nanmedian(antenna_scale)):.6f}")

    return {
        "stimulus": stimulus,
        "category": category,
        "mocap_path": path,
        "frames": frames.astype(np.int64),
        "x_relative": x_relative,
        "hand_prefix": hand_prefix,
        "hand_markers": hand_markers,
        "antenna_prefix": antenna_prefix,
        "antenna_markers": antenna_markers,
    }


# -----------------------------------------------------------------------------
# Stimulus discovery
# -----------------------------------------------------------------------------

def resolve_target_path(
    feature_dir: str,
    stimulus_candidates: Sequence[str],
    target_suffix: str,
    target_ext: str,
) -> Tuple[Optional[str], Optional[str], List[str]]:
    if target_ext == "auto":
        exts = [".npy", ".csv"]
    else:
        ext = target_ext if target_ext.startswith(".") else f".{target_ext}"
        exts = [ext]

    tried_paths = []
    for stimulus in stimulus_candidates:
        for ext in exts:
            candidate = os.path.join(feature_dir, f"{stimulus}_{target_suffix}{ext}")
            tried_paths.append(candidate)
            if os.path.exists(candidate):
                return stimulus, candidate, tried_paths

    return None, None, tried_paths


def discover_mocap_stimuli(
    feature_dir: str,
    mocap_dir: str,
    target_suffix: str,
    target_ext: str,
) -> List[Dict[str, str]]:
    mocap_paths = sorted(glob.glob(os.path.join(mocap_dir, "*_cleaned.csv")))

    if not mocap_paths:
        raise RuntimeError(f"No cleaned MOCAP CSV files found in: {mocap_dir}")

    pairs = []

    for mocap_path in mocap_paths:
        stem = os.path.splitext(os.path.basename(mocap_path))[0]
        stimulus_candidates = strip_known_mocap_suffixes(stem)

        stimulus, target_path, tried_paths = resolve_target_path(
            feature_dir=feature_dir,
            stimulus_candidates=stimulus_candidates,
            target_suffix=target_suffix,
            target_ext=target_ext,
        )

        if stimulus is None or target_path is None:
            print(
                f"Skipping {stem}: could not find target file in {feature_dir}. "
                f"Tried these exact paths: {tried_paths}"
            )
            continue

        category = category_from_stimulus(stimulus)
        if category is None:
            print(f"Skipping {stimulus}: name does not contain pitch or volume.")
            continue

        pairs.append(
            {
                "stimulus": stimulus,
                "category": category,
                "mocap_path": mocap_path,
                "target_path": target_path,
            }
        )

    if not pairs:
        raise RuntimeError(
            f"No valid cleaned MOCAP/audio pairs found. MOCAP dir: {mocap_dir}, "
            f"target dir: {feature_dir}."
        )

    return pairs


# -----------------------------------------------------------------------------
# Category range normalization
# -----------------------------------------------------------------------------

def compute_category_range_stats(
    prepared_rows: Sequence[Dict],
    epsilon: float,
) -> Dict[str, Dict[str, np.ndarray]]:
    stats = {}

    for category in sorted(set(row["category"] for row in prepared_rows)):
        xs = [row["x_relative"] for row in prepared_rows if row["category"] == category]
        if not xs:
            continue

        feature_dims = sorted(set(x.shape[1] for x in xs))
        if len(feature_dims) != 1:
            raise RuntimeError(
                f"MOCAP files in category {category!r} do not have the same feature dimension: "
                f"{feature_dims}. Check marker counts or use category-specific settings."
            )

        stacked = np.vstack(xs).astype(np.float32)
        x_min = np.nanmin(stacked, axis=0).astype(np.float32)
        x_max = np.nanmax(stacked, axis=0).astype(np.float32)
        x_range = (x_max - x_min).astype(np.float32)
        x_range[x_range < epsilon] = 1.0

        stats[category] = {
            "min": x_min,
            "max": x_max,
            "range": x_range,
        }

        print()
        print("Category range normalization stats")
        print(f"category:      {category}")
        print(f"n files:       {len(xs)}")
        print(f"feature dim:   {stacked.shape[1]}")
        print(f"rows:          {stacked.shape[0]}")
        print(f"global min:    {float(np.nanmin(stacked)):.6f}")
        print(f"global max:    {float(np.nanmax(stacked)):.6f}")

    return stats


def apply_range_normalization(
    x: np.ndarray,
    stats: Dict[str, np.ndarray],
    mode: str,
) -> np.ndarray:
    if mode == "none":
        return x.astype(np.float32)

    x_min = stats["min"].reshape(1, -1)
    x_range = stats["range"].reshape(1, -1)

    x_norm = (x - x_min) / x_range

    if mode == "zero_one":
        return x_norm.astype(np.float32)

    if mode == "minus_one_one":
        return (2.0 * x_norm - 1.0).astype(np.float32)

    raise RuntimeError(f"Unknown range normalization mode: {mode}")


def write_normalization_report(
    path: str,
    stats: Dict[str, Dict[str, np.ndarray]],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    rows = []
    for category, category_stats in stats.items():
        x_min = category_stats["min"]
        x_max = category_stats["max"]
        x_range = category_stats["range"]
        for idx in range(len(x_min)):
            rows.append(
                {
                    "category": category,
                    "feature_index": idx,
                    "min": float(x_min[idx]),
                    "max": float(x_max[idx]),
                    "range": float(x_range[idx]),
                }
            )

    write_csv(path, rows, ["category", "feature_index", "min", "max", "range"])


# -----------------------------------------------------------------------------
# Cleaning / FPS matching
# -----------------------------------------------------------------------------

def clean_target_array(target_arr: np.ndarray) -> np.ndarray:
    if target_arr.ndim == 2 and target_arr.shape[1] == 1:
        target_arr = target_arr[:, 0]

    if target_arr.ndim != 1:
        raise RuntimeError(f"Expected target array [frames] or [frames, 1], got {target_arr.shape}")

    print()
    print("Target cleaning")
    print(f"Original target frames: {len(target_arr)}")
    print(f"Target NaNs:            {int(np.isnan(target_arr).sum())}")

    return target_arr.astype(np.float32)


def resolved_max_nils_to_fill(max_nils_to_fill: int, n_features: int) -> int:
    if max_nils_to_fill >= 0:
        return max_nils_to_fill
    return max(1, int(round(0.20 * n_features)))


def clean_feature_array(feature_arr: np.ndarray, max_nils_to_fill: int) -> Tuple[np.ndarray, np.ndarray]:
    if feature_arr.ndim != 2:
        raise RuntimeError(f"Expected feature array [frames, features], got {feature_arr.shape}")

    effective_max_nils = resolved_max_nils_to_fill(max_nils_to_fill, feature_arr.shape[1])

    nil_count = np.isnan(feature_arr).sum(axis=1)
    keep_mask = nil_count <= effective_max_nils

    cleaned = feature_arr[keep_mask].copy()
    cleaned[np.isnan(cleaned)] = 0.0

    print()
    print("MOCAP feature cleaning")
    print(f"Original frames:        {len(feature_arr)}")
    print(f"Feature dimension:      {feature_arr.shape[1]}")
    print(f"Max NaNs filled/frame:  {effective_max_nils}")
    print(f"Kept frames:            {len(cleaned)}")
    print(f"Dropped frames:         {len(feature_arr) - len(cleaned)}")

    return cleaned.astype(np.float32), keep_mask


def resample_mocap_to_target_rate(
    feature_arr: np.ndarray,
    target_arr: np.ndarray,
    hand_fps: float,
    target_fps: float,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert high-rate MOCAP features to one feature row per target/CV frame.

    The previous implementation kept every MOCAP frame and mapped many MOCAP
    rows to the same audio target frame. With 360 Hz MOCAP and 60 Hz targets,
    this made the model see each target value roughly six times. This function
    collapses the MOCAP stream to the target frame rate before cleaning,
    splitting, and training.

    Returns:
        x_resampled:       [target_frames_with_mocap, features]
        y_resampled:       [target_frames_with_mocap]
        target_frames:     target/CV frame indices, typically 0..N-1 at 60 Hz
        source_counts:     number of MOCAP frames contributing to each target frame
    """
    if feature_arr.ndim != 2:
        raise RuntimeError(f"Expected feature_arr [frames, features], got {feature_arr.shape}")
    if target_arr.ndim != 1:
        raise RuntimeError(f"Expected target_arr [frames], got {target_arr.shape}")
    if mode not in {"mean", "nearest"}:
        raise RuntimeError(f"Unknown MOCAP resample mode: {mode}")

    raw_feature_len = len(feature_arr)
    raw_target_len = len(target_arr)
    n_features = feature_arr.shape[1]

    source_indices = np.arange(raw_feature_len, dtype=np.int64)
    source_times_s = source_indices.astype(np.float64) / float(hand_fps)
    target_idx_for_source = np.rint(source_times_s * float(target_fps)).astype(np.int64)

    valid_source = (target_idx_for_source >= 0) & (target_idx_for_source < raw_target_len)
    source_indices = source_indices[valid_source]
    target_idx_for_source = target_idx_for_source[valid_source]

    if len(source_indices) == 0:
        raise RuntimeError("No overlapping MOCAP frames could be mapped to target frames.")

    order = np.argsort(target_idx_for_source, kind="stable")
    sorted_source_indices = source_indices[order]
    sorted_target_indices = target_idx_for_source[order]

    target_frames, starts, source_counts = np.unique(
        sorted_target_indices,
        return_index=True,
        return_counts=True,
    )

    if mode == "nearest":
        target_times_s = target_frames.astype(np.float64) / float(target_fps)
        nearest_source_indices = np.rint(target_times_s * float(hand_fps)).astype(np.int64)
        nearest_source_indices = np.clip(nearest_source_indices, 0, raw_feature_len - 1)
        x_resampled = feature_arr[nearest_source_indices].astype(np.float32)
    else:
        x_resampled = np.full((len(target_frames), n_features), np.nan, dtype=np.float32)
        for out_idx, start, count in zip(range(len(target_frames)), starts, source_counts):
            source_group = sorted_source_indices[start:start + count]
            rows = feature_arr[source_group]
            finite = np.isfinite(rows)
            finite_counts = finite.sum(axis=0)
            finite_sums = np.where(finite, rows, 0.0).sum(axis=0, dtype=np.float64)

            averaged = np.full(n_features, np.nan, dtype=np.float64)
            np.divide(
                finite_sums,
                finite_counts,
                out=averaged,
                where=finite_counts > 0,
            )
            x_resampled[out_idx] = averaged.astype(np.float32)

    y_resampled = target_arr[target_frames].astype(np.float32)

    return (
        x_resampled.astype(np.float32),
        y_resampled.astype(np.float32),
        target_frames.astype(np.int64),
        source_counts.astype(np.int64),
    )


def build_fps_matched_arrays(
    feature_arr: np.ndarray,
    target_arr: np.ndarray,
    source_frames: np.ndarray,
    max_nils_to_fill: int,
    hand_fps: float,
    target_fps: float,
    mocap_resample_mode: str = "mean",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if hand_fps <= 0:
        raise RuntimeError(f"hand_fps must be > 0, got {hand_fps}")
    if target_fps <= 0:
        raise RuntimeError(f"target_fps must be > 0, got {target_fps}")

    raw_feature_len = len(feature_arr)
    raw_target_len = len(target_arr)

    if raw_feature_len == 0:
        raise RuntimeError("MOCAP feature array has 0 frames.")
    if raw_target_len == 0:
        raise RuntimeError("Target array has 0 frames.")
    if len(source_frames) != raw_feature_len:
        raise RuntimeError(
            f"source_frames length mismatch: {len(source_frames)} vs {raw_feature_len}"
        )

    print()
    print("=" * 80)
    print("Raw count check")
    print("=" * 80)
    print(f"Raw MOCAP frames:      {raw_feature_len}")
    print(f"Raw target frames:     {raw_target_len}")
    print(f"MOCAP FPS:             {hand_fps}")
    print(f"Target/CV FPS:         {target_fps}")
    print(f"MOCAP duration:        {raw_feature_len / hand_fps:.6f} s")
    print(f"Target duration:       {raw_target_len / target_fps:.6f} s")
    print(f"MOCAP / target ratio:  {raw_feature_len / max(raw_target_len, 1):.6f}")
    print(f"MOCAP resample mode:   {mocap_resample_mode}")
    print("=" * 80)

    target_arr = clean_target_array(target_arr)

    x_target_rate, y_target_rate, target_frames, source_counts = resample_mocap_to_target_rate(
        feature_arr=feature_arr,
        target_arr=target_arr,
        hand_fps=hand_fps,
        target_fps=target_fps,
        mode=mocap_resample_mode,
    )

    print()
    print("MOCAP resampled to target/CV frame rate")
    print(f"Resampled feature shape:   {x_target_rate.shape}")
    print(f"Resampled target shape:    {y_target_rate.shape}")
    print(f"Target-frame range:        {int(target_frames[0])} -> {int(target_frames[-1])}")
    print(f"MOCAP frames/CV frame:     mean={float(np.mean(source_counts)):.3f}, "
          f"min={int(np.min(source_counts))}, max={int(np.max(source_counts))}")

    x_clean, keep_mask = clean_feature_array(x_target_rate, max_nils_to_fill)
    y_clean = y_target_rate[keep_mask]
    frames_clean = target_frames[keep_mask]

    valid_y = np.isfinite(y_clean)
    x_clean = x_clean[valid_y]
    y_clean = y_clean[valid_y]
    frames_clean = frames_clean[valid_y]

    if len(x_clean) == 0:
        raise RuntimeError("Final arrays have 0 rows after cleaning.")

    print()
    print("Final 60 Hz CV data")
    print(f"x shape:       {x_clean.shape}")
    print(f"y shape:       {y_clean.shape}")
    print(f"frames shape:  {frames_clean.shape}")
    print(f"Final samples: {len(x_clean)}")
    print(f"First CV frame: {int(frames_clean[0])}")
    print(f"Last CV frame:  {int(frames_clean[-1])}")

    if hand_fps == target_fps and raw_feature_len == raw_target_len:
        print("One-to-one frame mapping detected.")

    return x_clean.astype(np.float32), y_clean.astype(np.float32), frames_clean.astype(np.int64)


# -----------------------------------------------------------------------------
# Dataset / splits
# -----------------------------------------------------------------------------

class MocapSequenceDataset(Dataset):
    def __init__(
        self,
        x_arr: np.ndarray,
        y_arr: np.ndarray,
        frames: np.ndarray,
        end_indices: np.ndarray,
        seq_len: int,
        x_mean: Optional[np.ndarray] = None,
        x_std: Optional[np.ndarray] = None,
    ):
        super().__init__()

        self.x = np.asarray(x_arr, dtype=np.float32)
        self.y = np.asarray(y_arr, dtype=np.float32).reshape(-1)
        self.frames = np.asarray(frames, dtype=np.int64).reshape(-1)
        self.end_indices = np.asarray(end_indices, dtype=np.int64).reshape(-1)
        self.seq_len = int(seq_len)

        if self.x.ndim != 2:
            raise RuntimeError(f"Expected x_arr [frames, features], got {self.x.shape}")
        if len(self.x) != len(self.y):
            raise RuntimeError(f"x/y length mismatch: {len(self.x)} vs {len(self.y)}")
        if len(self.x) != len(self.frames):
            raise RuntimeError(f"x/frames length mismatch: {len(self.x)} vs {len(self.frames)}")
        if self.seq_len <= 0:
            raise RuntimeError(f"seq_len must be > 0, got {self.seq_len}")

        self.end_indices = self.end_indices[self.end_indices >= self.seq_len - 1]
        self.end_indices = self.end_indices[self.end_indices < len(self.x)]

        self.x_mean = None
        self.x_std = None
        if x_mean is not None and x_std is not None:
            self.set_feature_stats(x_mean, x_std)

    def set_feature_stats(self, x_mean: np.ndarray, x_std: np.ndarray) -> None:
        self.x_mean = np.asarray(x_mean, dtype=np.float32).reshape(1, -1)
        self.x_std = np.asarray(x_std, dtype=np.float32).reshape(1, -1)
        self.x_std[self.x_std < 1e-6] = 1.0

    def __len__(self) -> int:
        return len(self.end_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        end = int(self.end_indices[idx])
        start = end - self.seq_len + 1

        x = self.x[start:end + 1].copy()
        if self.x_mean is not None and self.x_std is not None:
            x = (x - self.x_mean) / self.x_std

        y = np.float32(self.y[end])
        frame = int(self.frames[end])

        return {
            "x": torch.from_numpy(x),
            "y": torch.tensor(y, dtype=torch.float32),
            "frame": torch.tensor(frame, dtype=torch.long),
        }


class MultiMocapSequenceDataset(Dataset):
    """
    Dataset for category-level aggregate training.

    Each sample is a pair (row_index, end_index). The sequence is sliced only
    inside that stimulus row, so a seq_len=5 window can never cross from one
    stimulus into another.
    """
    def __init__(
        self,
        rows: Sequence[Dict],
        samples: Sequence[Tuple[int, int]],
        seq_len: int,
        x_mean: Optional[np.ndarray] = None,
        x_std: Optional[np.ndarray] = None,
    ):
        super().__init__()

        self.rows = list(rows)
        self.samples = [(int(row_idx), int(end_idx)) for row_idx, end_idx in samples]
        self.seq_len = int(seq_len)

        if self.seq_len <= 0:
            raise RuntimeError(f"seq_len must be > 0, got {self.seq_len}")
        if not self.rows:
            raise RuntimeError("MultiMocapSequenceDataset received no rows.")

        input_dims = sorted(set(int(row["x_arr"].shape[1]) for row in self.rows))
        if len(input_dims) != 1:
            raise RuntimeError(
                "Aggregate dataset contains incompatible feature dimensions: "
                f"{input_dims}. Pitch and volume must be trained in separate aggregate stages."
            )

        clean_samples = []
        for row_idx, end_idx in self.samples:
            if row_idx < 0 or row_idx >= len(self.rows):
                continue
            x = self.rows[row_idx]["x_arr"]
            if end_idx >= self.seq_len - 1 and end_idx < len(x):
                clean_samples.append((row_idx, end_idx))
        self.samples = clean_samples

        self.x_mean = None
        self.x_std = None
        if x_mean is not None and x_std is not None:
            self.set_feature_stats(x_mean, x_std)

    def set_feature_stats(self, x_mean: np.ndarray, x_std: np.ndarray) -> None:
        self.x_mean = np.asarray(x_mean, dtype=np.float32).reshape(1, -1)
        self.x_std = np.asarray(x_std, dtype=np.float32).reshape(1, -1)
        self.x_std[self.x_std < 1e-6] = 1.0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row_idx, end = self.samples[idx]
        row = self.rows[row_idx]
        start = end - self.seq_len + 1

        x = row["x_arr"][start:end + 1].copy()
        if self.x_mean is not None and self.x_std is not None:
            x = (x - self.x_mean) / self.x_std

        y = np.float32(row["y_arr"][end])
        frame = int(row["frames_cv"][end])

        return {
            "x": torch.from_numpy(x),
            "y": torch.tensor(y, dtype=torch.float32),
            "frame": torch.tensor(frame, dtype=torch.long),
            "row_index": torch.tensor(row_idx, dtype=torch.long),
        }


@torch.no_grad()
def compute_feature_stats(dataset: Dataset, batch_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_sum = None
    total_sq_sum = None
    total_count = 0

    for batch in loader:
        x = batch["x"].float()
        x = x.reshape(-1, x.shape[-1])

        if total_sum is None:
            total_sum = x.sum(dim=0)
            total_sq_sum = (x ** 2).sum(dim=0)
        else:
            total_sum += x.sum(dim=0)
            total_sq_sum += (x ** 2).sum(dim=0)

        total_count += x.shape[0]

    if total_count == 0:
        raise RuntimeError("Cannot compute feature stats: dataset has 0 samples.")

    mean = total_sum / total_count
    var = total_sq_sum / total_count - mean ** 2
    std = torch.sqrt(torch.clamp(var, min=1e-8))

    return mean.numpy().astype(np.float32), std.numpy().astype(np.float32)


def make_kfold_end_indices(n_frames: int, seq_len: int, n_folds: int, split: str, seed: int):
    if n_folds < 2:
        raise RuntimeError(f"n_folds must be >= 2, got {n_folds}")
    if n_frames < seq_len:
        raise RuntimeError(f"Not enough frames for seq_len={seq_len}: got {n_frames}")

    valid_ends = np.arange(seq_len - 1, n_frames, dtype=np.int64)

    if len(valid_ends) < n_folds:
        raise RuntimeError(f"Cannot create {n_folds} folds from only {len(valid_ends)} samples")

    if split == "random":
        rng = np.random.default_rng(seed)
        rng.shuffle(valid_ends)
    elif split == "chronological":
        pass
    else:
        raise RuntimeError(f"Unknown split mode: {split}")

    folds = np.array_split(valid_ends, n_folds)
    out = []

    for fold_idx in range(n_folds):
        test_ends = np.sort(folds[fold_idx])
        train_ends = np.sort(np.concatenate([folds[i] for i in range(n_folds) if i != fold_idx]))
        out.append((train_ends, test_ends))

    return out


def split_train_val_end_indices(
    train_ends: np.ndarray,
    val_ratio_within_train: float,
    seed: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    train_ends = np.asarray(train_ends, dtype=np.int64)

    if val_ratio_within_train <= 0.0:
        return np.sort(train_ends), None
    if val_ratio_within_train >= 1.0:
        raise RuntimeError("--val-ratio-within-train must be < 1.0")

    rng = np.random.default_rng(seed)
    shuffled = np.array(train_ends, copy=True)
    rng.shuffle(shuffled)

    n_val = max(1, int(round(len(shuffled) * val_ratio_within_train)))
    val_ends = np.sort(shuffled[:n_val])
    clean_train_ends = np.sort(shuffled[n_val:])

    if len(clean_train_ends) == 0:
        raise RuntimeError("Training split has 0 samples after validation split.")

    return clean_train_ends, val_ends


def sort_samples(samples: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return sorted((int(row_idx), int(end_idx)) for row_idx, end_idx in samples)


def make_balanced_multi_kfold_samples(
    rows: Sequence[Dict],
    seq_len: int,
    n_folds: int,
    split: str,
    seed: int,
) -> List[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]:
    """
    Build balanced category-level folds.

    For every stimulus in the category, this creates its own K folds. Fold k
    then contains a test partition from every stimulus, and a training partition
    from every stimulus. This avoids folds dominated by a single recording.
    """
    if n_folds < 2:
        raise RuntimeError(f"n_folds must be >= 2, got {n_folds}")
    if split not in {"chronological", "random"}:
        raise RuntimeError(f"Unknown split mode: {split}")

    per_row_folds = []
    for row_idx, row in enumerate(rows):
        n_frames = len(row["x_arr"])
        stimulus = row.get("stimulus", f"row_{row_idx}")

        if n_frames < seq_len:
            raise RuntimeError(
                f"Not enough frames for seq_len={seq_len} in stimulus {stimulus}: got {n_frames}"
            )

        valid_ends = np.arange(seq_len - 1, n_frames, dtype=np.int64)
        if len(valid_ends) < n_folds:
            raise RuntimeError(
                f"Cannot create {n_folds} folds from only {len(valid_ends)} samples "
                f"in stimulus {stimulus}"
            )

        if split == "random":
            rng = np.random.default_rng(seed + 1009 * (row_idx + 1))
            rng.shuffle(valid_ends)

        folds = np.array_split(valid_ends, n_folds)
        per_row_folds.append(folds)

    out = []
    for fold_idx in range(n_folds):
        train_samples = []
        test_samples = []
        for row_idx, folds in enumerate(per_row_folds):
            test_ends = np.sort(folds[fold_idx])
            train_ends = np.sort(np.concatenate([folds[i] for i in range(n_folds) if i != fold_idx]))
            train_samples.extend((row_idx, int(end_idx)) for end_idx in train_ends)
            test_samples.extend((row_idx, int(end_idx)) for end_idx in test_ends)
        out.append((sort_samples(train_samples), sort_samples(test_samples)))

    return out


def split_multi_train_val_samples(
    train_samples: Sequence[Tuple[int, int]],
    val_ratio_within_train: float,
    seed: int,
) -> Tuple[List[Tuple[int, int]], Optional[List[Tuple[int, int]]]]:
    if val_ratio_within_train <= 0.0:
        return sort_samples(train_samples), None
    if val_ratio_within_train >= 1.0:
        raise RuntimeError("--val-ratio-within-train must be < 1.0")

    grouped: Dict[int, List[Tuple[int, int]]] = {}
    for row_idx, end_idx in train_samples:
        grouped.setdefault(int(row_idx), []).append((int(row_idx), int(end_idx)))

    rng = np.random.default_rng(seed)
    clean_train = []
    val = []

    for row_idx in sorted(grouped):
        group = grouped[row_idx]
        order = np.arange(len(group), dtype=np.int64)
        rng.shuffle(order)

        if len(group) <= 1:
            clean_train.extend(group)
            continue

        n_val = max(1, int(round(len(group) * val_ratio_within_train)))
        n_val = min(n_val, len(group) - 1)

        val_indices = set(int(idx) for idx in order[:n_val])
        for i, sample in enumerate(group):
            if i in val_indices:
                val.append(sample)
            else:
                clean_train.append(sample)

    if not clean_train:
        raise RuntimeError("Aggregate training split has 0 samples after validation split.")

    return sort_samples(clean_train), sort_samples(val) if val else None


def print_multi_target_stats(name: str, rows: Sequence[Dict], samples: Sequence[Tuple[int, int]]) -> None:
    values = []
    counts: Dict[str, int] = {}

    for row_idx, end_idx in samples:
        row = rows[int(row_idx)]
        values.append(float(row["y_arr"][int(end_idx)]))
        stimulus = row.get("stimulus", f"row_{row_idx}")
        counts[stimulus] = counts.get(stimulus, 0) + 1

    values_arr = np.asarray(values, dtype=np.float32)
    if len(values_arr) == 0:
        raise RuntimeError(f"{name} aggregate split has 0 samples.")

    print()
    print(f"{name} aggregate stats")
    print(f"samples: {len(values_arr)}")
    print(f"y min:   {np.min(values_arr):.6f}")
    print(f"y max:   {np.max(values_arr):.6f}")
    print(f"y mean:  {np.mean(values_arr):.6f}")
    print(f"y std:   {np.std(values_arr):.6f}")
    print("samples by stimulus:")
    for stimulus in sorted(counts):
        print(f"  {stimulus}: {counts[stimulus]}")


def print_target_stats(name: str, y: np.ndarray, end_indices: np.ndarray) -> None:
    values = y[np.asarray(end_indices, dtype=np.int64)]
    print()
    print(f"{name} stats")
    print(f"samples: {len(values)}")
    print(f"y min:   {np.min(values):.6f}")
    print(f"y max:   {np.max(values):.6f}")
    print(f"y mean:  {np.mean(values):.6f}")
    print(f"y std:   {np.std(values):.6f}")


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


# -----------------------------------------------------------------------------
# Training / evaluation
# -----------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad(set_to_none=True)
        y_hat = model(x).reshape_as(y)
        loss = loss_fn(y_hat, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_count += x.size(0)

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        y_hat = model(x).reshape_as(y)
        loss = loss_fn(y_hat, y)

        total_loss += loss.item() * x.size(0)
        total_count += x.size(0)

    return total_loss / max(total_count, 1)


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()

    all_frames = []
    all_y_true = []
    all_y_pred = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].cpu().numpy()
        frames = batch["frame"].cpu().numpy()
        y_hat = model(x).reshape_as(batch["y"].to(device)).cpu().numpy()

        all_frames.append(frames)
        all_y_true.append(y)
        all_y_pred.append(y_hat)

    if not all_frames:
        return np.array([]), np.array([]), np.array([])

    frames = np.concatenate(all_frames)
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)

    order = np.argsort(frames)
    return frames[order], y_true[order], y_pred[order]


@torch.no_grad()
def collect_predictions_multi(model, loader, rows: Sequence[Dict], device):
    model.eval()

    all_row_indices = []
    all_frames = []
    all_y_true = []
    all_y_pred = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].cpu().numpy()
        frames = batch["frame"].cpu().numpy()
        row_indices = batch["row_index"].cpu().numpy()
        y_hat = model(x).reshape_as(batch["y"].to(device)).cpu().numpy()

        all_row_indices.append(row_indices)
        all_frames.append(frames)
        all_y_true.append(y)
        all_y_pred.append(y_hat)

    if not all_frames:
        return np.array([]), np.array([]), np.array([]), np.array([])

    row_indices = np.concatenate(all_row_indices).astype(np.int64)
    frames = np.concatenate(all_frames)
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    stimuli = np.asarray([rows[int(idx)]["stimulus"] for idx in row_indices], dtype=object)

    order = np.lexsort((frames, stimuli.astype(str)))
    return stimuli[order], frames[order], y_true[order], y_pred[order]


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {"mse": float("nan"), "mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}

    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(math.sqrt(mse))

    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom < 1e-12:
        r2 = float("nan")
    else:
        r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)

    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}


# -----------------------------------------------------------------------------
# Plotting / output
# -----------------------------------------------------------------------------

def save_predictions(frames, y_true, y_pred, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data = np.column_stack([frames, y_true, y_pred, y_pred - y_true])
    np.savetxt(
        out_path,
        data,
        delimiter=",",
        header="Frame,GroundTruth,Prediction,Error",
        comments="",
        fmt=["%d", "%.8f", "%.8f", "%.8f"],
    )
    print(f"Prediction CSV saved to: {out_path}")


def save_predictions_multi(stimuli, frames, y_true, y_pred, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Stimulus", "Frame", "GroundTruth", "Prediction", "Error"])
        for stimulus, frame, gt, pred in zip(stimuli, frames, y_true, y_pred):
            writer.writerow([
                str(stimulus),
                int(frame),
                f"{float(gt):.8f}",
                f"{float(pred):.8f}",
                f"{float(pred - gt):.8f}",
            ])
    print(f"Prediction CSV saved to: {out_path}")


def write_csv(path: str, rows: Sequence[Dict], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"CSV saved to: {path}")


def save_data_overview_plots(x_arr: np.ndarray, y_arr: np.ndarray, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(np.arange(len(y_arr)), y_arr)
    ax.set_title("Full target signal")
    ax.set_xlabel("Aligned MOCAP-frame index")
    ax.set_ylabel("Target")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "full_target.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_arr, bins=50, alpha=0.8)
    ax.set_title("Target distribution")
    ax.set_xlabel("Target")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "target_histogram.png"), dpi=150)
    plt.close(fig)

    y_norm = (y_arr - np.mean(y_arr)) / (np.std(y_arr) + 1e-8)
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(y_norm, label="target normalized", linewidth=1.8)

    feature_indices = list(range(min(3, x_arr.shape[1])))
    for feature_idx in feature_indices:
        x_feat = x_arr[:, feature_idx]
        x_feat = (x_feat - np.mean(x_feat)) / (np.std(x_feat) + 1e-8)
        ax.plot(x_feat, label=f"mocap feature {feature_idx} normalized", alpha=0.7)

    ax.set_title("Target vs selected MOCAP features")
    ax.set_xlabel("Aligned MOCAP-frame index")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "target_vs_mocap_features.png"), dpi=150)
    plt.close(fig)


def plot_loss_curve(history: Sequence[Dict[str, float]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not history:
        return

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_mse"] for row in history]
    monitor_loss = [row["monitor_mse"] for row in history]
    monitor_name = history[0].get("monitor_name", "monitor")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_loss, label="train MSE", linewidth=2.0)
    ax.plot(epochs, monitor_loss, label=f"{monitor_name} MSE", linewidth=2.0)
    ax.set_title("Training curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_lr_curve(history: Sequence[Dict[str, float]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not history:
        return

    epochs = [row["epoch"] for row in history]
    lr = [row["lr"] for row in history]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, lr, linewidth=2.0)
    ax.set_title("Learning rate schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_predictions(frames, y_true, y_pred, out_path: str, chunk_size: int, title_prefix: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if len(frames) == 0:
        return

    metrics = compute_regression_metrics(y_true, y_pred)
    chunk_size = min(chunk_size, len(frames))

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(frames[:chunk_size], y_true[:chunk_size], linewidth=2.0, label="Ground truth")
    ax.plot(frames[:chunk_size], y_pred[:chunk_size], linewidth=2.0, linestyle="--", label="Prediction")
    ax.set_title(
        f"{title_prefix} | MSE={metrics['mse']:.6f} | "
        f"RMSE={metrics['rmse']:.6f} | MAE={metrics['mae']:.6f} | R2={metrics['r2']:.4f}"
    )
    ax.set_xlabel("Frame")
    ax.set_ylabel("Target")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_predictions_by_sample_order(y_true, y_pred, out_path: str, chunk_size: int, title_prefix: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if len(y_true) == 0:
        return

    metrics = compute_regression_metrics(y_true, y_pred)
    chunk_size = min(chunk_size, len(y_true))
    x_axis = np.arange(chunk_size)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(x_axis, y_true[:chunk_size], linewidth=2.0, label="Ground truth")
    ax.plot(x_axis, y_pred[:chunk_size], linewidth=2.0, linestyle="--", label="Prediction")
    ax.set_title(
        f"{title_prefix} | MSE={metrics['mse']:.6f} | "
        f"RMSE={metrics['rmse']:.6f} | MAE={metrics['mae']:.6f} | R2={metrics['r2']:.4f}"
    )
    ax.set_xlabel("Aggregate test-sample order")
    ax.set_ylabel("Target")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scatter(y_true, y_pred, out_path: str, title_prefix: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if len(y_true) == 0:
        return

    metrics = compute_regression_metrics(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.45, s=18)

    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=2.0)

    ax.set_title(
        f"{title_prefix}\nMSE={metrics['mse']:.6f} | "
        f"RMSE={metrics['rmse']:.6f} | MAE={metrics['mae']:.6f} | R2={metrics['r2']:.4f}"
    )
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Prediction")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_error_histogram(y_true, y_pred, out_path: str, title_prefix: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if len(y_true) == 0:
        return

    err = y_pred - y_true
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(err, bins=50, alpha=0.85)
    ax.set_title(f"{title_prefix} error distribution")
    ax.set_xlabel("Prediction error")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_summary_metric(summary_rows: Sequence[Dict], metric: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not summary_rows:
        return

    labels = [f"{row['stimulus']}\n{row['cycle']}" for row in summary_rows]
    means = np.asarray([row[f"{metric}_mean"] for row in summary_rows], dtype=np.float64)
    stds = np.asarray([row[f"{metric}_std"] for row in summary_rows], dtype=np.float64)
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(12, 0.75 * len(labels)), 6))
    ax.bar(x, means, yerr=stds, capsize=3)
    ax.set_title(f"Cross-validation summary: {metric}")
    ax.set_xlabel("Stimulus / cycle")
    ax.set_ylabel(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cycle_comparison(summary_rows: Sequence[Dict], metric: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not summary_rows:
        return

    stimuli = sorted(set(row["stimulus"] for row in summary_rows))
    cycles = sorted(set(row["cycle"] for row in summary_rows))
    if len(cycles) < 2:
        return

    by_key = {(row["stimulus"], row["cycle"]): row for row in summary_rows}
    x = np.arange(len(stimuli))
    width = 0.8 / len(cycles)

    fig, ax = plt.subplots(figsize=(max(12, 0.7 * len(stimuli)), 6))
    for i, cycle in enumerate(cycles):
        values = []
        errors = []
        for stimulus in stimuli:
            row = by_key.get((stimulus, cycle))
            if row is None:
                values.append(np.nan)
                errors.append(0.0)
            else:
                values.append(row[f"{metric}_mean"])
                errors.append(row[f"{metric}_std"])
        offset = (i - (len(cycles) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, yerr=errors, capsize=3, label=cycle)

    ax.set_title(f"Frame vs sequence comparison: {metric}")
    ax.set_xlabel("Stimulus")
    ax.set_ylabel(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(stimuli, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_summary_plots(summary_rows: Sequence[Dict], output_dir: str) -> None:
    plot_dir = os.path.join(output_dir, "summary_plots")
    metrics = ["test_mse", "test_rmse", "test_mae", "test_r2"]
    for metric in metrics:
        plot_summary_metric(summary_rows, metric, os.path.join(plot_dir, f"{metric}_summary.png"))
        plot_cycle_comparison(summary_rows, metric, os.path.join(plot_dir, f"{metric}_cycle_comparison.png"))


# -----------------------------------------------------------------------------
# Fold / cycle runner
# -----------------------------------------------------------------------------

def make_model(args, device, input_dim: int):
    return HandNet(
        input_dim=input_dim,
        coord_mlp_dim=args.coord_mlp_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)


def train_one_fold(
    cycle_name: str,
    seq_len: int,
    stimulus: str,
    category: str,
    fold_idx: int,
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    frames: np.ndarray,
    train_ends: np.ndarray,
    test_ends: np.ndarray,
    args,
    device,
) -> Dict:
    print()
    print("=" * 80)
    print(
        f"Cycle: {cycle_name} | seq_len={seq_len} | Category: {category} | "
        f"Stimulus: {stimulus} | Fold {fold_idx + 1}/{args.n_folds}"
    )
    print("=" * 80)

    train_ends, val_ends = split_train_val_end_indices(
        train_ends=train_ends,
        val_ratio_within_train=args.val_ratio_within_train,
        seed=args.seed + fold_idx,
    )

    print_target_stats("Train", y_arr, train_ends)
    if val_ends is not None:
        print_target_stats("Val", y_arr, val_ends)
    print_target_stats("Test", y_arr, test_ends)

    train_ds = MocapSequenceDataset(x_arr, y_arr, frames, train_ends, seq_len)
    test_ds = MocapSequenceDataset(x_arr, y_arr, frames, test_ends, seq_len)

    if len(train_ds) == 0:
        raise RuntimeError("Training dataset has 0 samples.")
    if len(test_ds) == 0:
        raise RuntimeError("Test dataset has 0 samples.")

    input_dim = int(x_arr.shape[1])

    if args.use_fold_standardization:
        x_mean, x_std = compute_feature_stats(train_ds, batch_size=args.batch_size)
    else:
        x_mean = np.zeros(input_dim, dtype=np.float32)
        x_std = np.ones(input_dim, dtype=np.float32)

    train_ds.set_feature_stats(x_mean, x_std)
    test_ds.set_feature_stats(x_mean, x_std)

    if val_ends is not None:
        val_ds = MocapSequenceDataset(x_arr, y_arr, frames, val_ends, seq_len)
        val_ds.set_feature_stats(x_mean, x_std)
    else:
        val_ds = None

    print()
    print("Dataset sizes")
    print(f"Input dim:     {input_dim}")
    print(f"Train samples: {len(train_ds)}")
    if val_ds is not None:
        print(f"Val samples:   {len(val_ds)}")
    print(f"Test samples:  {len(test_ds)}")

    train_loader = build_loader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers)
    monitor_loader = build_loader(val_ds if val_ds is not None else train_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = build_loader(test_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = make_model(args, device, input_dim=input_dim)

    print()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )
    loss_fn = nn.MSELoss()

    fold_dir = os.path.join(args.output_dir, cycle_name, stimulus, f"fold_{fold_idx + 1:02d}")
    checkpoint_path = os.path.join(fold_dir, "checkpoints", "best_model.pt")
    plot_dir = os.path.join(fold_dir, "plots")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    best_monitor_loss = float("inf")
    monitor_name = "val" if val_ds is not None else "train"
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        monitor_loss = evaluate(model, monitor_loader, loss_fn, device)

        old_lr = get_lr(optimizer)
        scheduler.step(monitor_loss)
        new_lr = get_lr(optimizer)

        history.append(
            {
                "epoch": epoch,
                "train_mse": train_loss,
                "monitor_mse": monitor_loss,
                "monitor_name": monitor_name,
                "lr": new_lr,
            }
        )

        print(
            f"Epoch {epoch:03d} | train MSE: {train_loss:.6f} | "
            f"{monitor_name} MSE: {monitor_loss:.6f} | lr: {new_lr:.8f}"
        )

        if new_lr < old_lr:
            print(f"Scheduler reduced LR: {old_lr:.8f} -> {new_lr:.8f}")

        if monitor_loss < best_monitor_loss:
            best_monitor_loss = monitor_loss
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "cycle": cycle_name,
                "seq_len": seq_len,
                "stimulus": stimulus,
                "category": category,
                "input_dim": input_dim,
                "fold": fold_idx + 1,
                "train_end_indices": train_ends,
                "val_end_indices": val_ends,
                "test_end_indices": test_ends,
                "x_mean": torch.tensor(x_mean, dtype=torch.float32),
                "x_std": torch.tensor(x_std, dtype=torch.float32),
                "best_monitor_loss": float(best_monitor_loss),
                "monitor_name": monitor_name,
            }
            torch.save(checkpoint, checkpoint_path)

    history_csv_path = os.path.join(fold_dir, "training_history.csv")
    write_csv(
        history_csv_path,
        history,
        fieldnames=["epoch", "train_mse", "monitor_mse", "monitor_name", "lr"],
    )

    print()
    print(f"Best {monitor_name} MSE: {best_monitor_loss:.6f}")
    print(f"Saved best model to: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    frames_out, y_true, y_pred = collect_predictions(model, test_loader, device)
    metrics = compute_regression_metrics(y_true, y_pred)

    print()
    print("Test results")
    print(f"MSE:  {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE:  {metrics['mae']:.6f}")
    print(f"R2:   {metrics['r2']:.6f}")

    prediction_csv_path = os.path.join(fold_dir, "test_predictions.csv")
    save_predictions(frames_out, y_true, y_pred, prediction_csv_path)

    if args.save_plots:
        title_prefix = f"{cycle_name} | {stimulus} | fold {fold_idx + 1:02d}"
        plot_loss_curve(history, os.path.join(plot_dir, "loss_curve.png"))
        plot_lr_curve(history, os.path.join(plot_dir, "lr_curve.png"))
        plot_predictions(
            frames_out,
            y_true,
            y_pred,
            os.path.join(plot_dir, "test_predictions_chunk.png"),
            args.plot_chunk_size,
            title_prefix,
        )
        plot_predictions(
            frames_out,
            y_true,
            y_pred,
            os.path.join(plot_dir, "test_predictions_full.png"),
            len(frames_out),
            title_prefix,
        )
        plot_scatter(y_true, y_pred, os.path.join(plot_dir, "scatter.png"), title_prefix)
        plot_error_histogram(y_true, y_pred, os.path.join(plot_dir, "error_histogram.png"), title_prefix)

    return {
        "cycle": cycle_name,
        "seq_len": seq_len,
        "category": category,
        "stimulus": stimulus,
        "fold": fold_idx + 1,
        "input_dim": input_dim,
        "n_train_samples": len(train_ds),
        "n_val_samples": 0 if val_ds is None else len(val_ds),
        "n_test_samples": len(test_ds),
        "best_monitor_mse": best_monitor_loss,
        "monitor_name": monitor_name,
        "test_mse": metrics["mse"],
        "test_rmse": metrics["rmse"],
        "test_mae": metrics["mae"],
        "test_r2": metrics["r2"],
        "checkpoint_path": checkpoint_path,
        "prediction_csv_path": prediction_csv_path,
        "history_csv_path": history_csv_path,
    }


def train_one_category_aggregate_fold(
    cycle_name: str,
    seq_len: int,
    aggregate_name: str,
    category: str,
    fold_idx: int,
    rows: Sequence[Dict],
    train_samples: Sequence[Tuple[int, int]],
    test_samples: Sequence[Tuple[int, int]],
    args,
    device,
) -> Dict:
    print()
    print("=" * 80)
    print(
        f"Aggregate cycle: {cycle_name} | seq_len={seq_len} | Category: {category} | "
        f"Stage: {aggregate_name} | Fold {fold_idx + 1}/{args.n_folds}"
    )
    print("=" * 80)

    train_samples, val_samples = split_multi_train_val_samples(
        train_samples=train_samples,
        val_ratio_within_train=args.val_ratio_within_train,
        seed=args.seed + fold_idx,
    )

    print_multi_target_stats("Train", rows, train_samples)
    if val_samples is not None:
        print_multi_target_stats("Val", rows, val_samples)
    print_multi_target_stats("Test", rows, test_samples)

    train_ds = MultiMocapSequenceDataset(rows, train_samples, seq_len)
    test_ds = MultiMocapSequenceDataset(rows, test_samples, seq_len)

    if len(train_ds) == 0:
        raise RuntimeError("Aggregate training dataset has 0 samples.")
    if len(test_ds) == 0:
        raise RuntimeError("Aggregate test dataset has 0 samples.")

    input_dim = int(rows[0]["x_arr"].shape[1])

    if args.use_fold_standardization:
        x_mean, x_std = compute_feature_stats(train_ds, batch_size=args.batch_size)
    else:
        x_mean = np.zeros(input_dim, dtype=np.float32)
        x_std = np.ones(input_dim, dtype=np.float32)

    train_ds.set_feature_stats(x_mean, x_std)
    test_ds.set_feature_stats(x_mean, x_std)

    if val_samples is not None:
        val_ds = MultiMocapSequenceDataset(rows, val_samples, seq_len)
        val_ds.set_feature_stats(x_mean, x_std)
    else:
        val_ds = None

    print()
    print("Aggregate dataset sizes")
    print(f"Input dim:     {input_dim}")
    print(f"Stimuli:       {len(rows)}")
    print(f"Train samples: {len(train_ds)}")
    if val_ds is not None:
        print(f"Val samples:   {len(val_ds)}")
    print(f"Test samples:  {len(test_ds)}")

    train_loader = build_loader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers)
    monitor_loader = build_loader(val_ds if val_ds is not None else train_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = build_loader(test_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = make_model(args, device, input_dim=input_dim)

    print()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )
    loss_fn = nn.MSELoss()

    fold_dir = os.path.join(args.output_dir, "aggregate", cycle_name, aggregate_name, f"fold_{fold_idx + 1:02d}")
    checkpoint_path = os.path.join(fold_dir, "checkpoints", "best_model.pt")
    plot_dir = os.path.join(fold_dir, "plots")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    best_monitor_loss = float("inf")
    monitor_name = "val" if val_ds is not None else "train"
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        monitor_loss = evaluate(model, monitor_loader, loss_fn, device)

        old_lr = get_lr(optimizer)
        scheduler.step(monitor_loss)
        new_lr = get_lr(optimizer)

        history.append(
            {
                "epoch": epoch,
                "train_mse": train_loss,
                "monitor_mse": monitor_loss,
                "monitor_name": monitor_name,
                "lr": new_lr,
            }
        )

        print(
            f"Epoch {epoch:03d} | train MSE: {train_loss:.6f} | "
            f"{monitor_name} MSE: {monitor_loss:.6f} | lr: {new_lr:.8f}"
        )

        if new_lr < old_lr:
            print(f"Scheduler reduced LR: {old_lr:.8f} -> {new_lr:.8f}")

        if monitor_loss < best_monitor_loss:
            best_monitor_loss = monitor_loss
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "cycle": cycle_name,
                "seq_len": seq_len,
                "stimulus": aggregate_name,
                "category": category,
                "aggregate_stimuli": [row["stimulus"] for row in rows],
                "input_dim": input_dim,
                "fold": fold_idx + 1,
                "train_samples": np.asarray(train_samples, dtype=np.int64),
                "val_samples": None if val_samples is None else np.asarray(val_samples, dtype=np.int64),
                "test_samples": np.asarray(test_samples, dtype=np.int64),
                "x_mean": torch.tensor(x_mean, dtype=torch.float32),
                "x_std": torch.tensor(x_std, dtype=torch.float32),
                "best_monitor_loss": float(best_monitor_loss),
                "monitor_name": monitor_name,
            }
            torch.save(checkpoint, checkpoint_path)

    history_csv_path = os.path.join(fold_dir, "training_history.csv")
    write_csv(
        history_csv_path,
        history,
        fieldnames=["epoch", "train_mse", "monitor_mse", "monitor_name", "lr"],
    )

    print()
    print(f"Best {monitor_name} MSE: {best_monitor_loss:.6f}")
    print(f"Saved best model to: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    stimuli_out, frames_out, y_true, y_pred = collect_predictions_multi(model, test_loader, rows, device)
    metrics = compute_regression_metrics(y_true, y_pred)

    print()
    print("Aggregate test results")
    print(f"MSE:  {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE:  {metrics['mae']:.6f}")
    print(f"R2:   {metrics['r2']:.6f}")

    prediction_csv_path = os.path.join(fold_dir, "test_predictions.csv")
    save_predictions_multi(stimuli_out, frames_out, y_true, y_pred, prediction_csv_path)

    if args.save_plots:
        title_prefix = f"aggregate | {cycle_name} | {aggregate_name} | fold {fold_idx + 1:02d}"
        plot_loss_curve(history, os.path.join(plot_dir, "loss_curve.png"))
        plot_lr_curve(history, os.path.join(plot_dir, "lr_curve.png"))
        plot_predictions_by_sample_order(
            y_true,
            y_pred,
            os.path.join(plot_dir, "test_predictions_chunk.png"),
            args.plot_chunk_size,
            title_prefix,
        )
        plot_predictions_by_sample_order(
            y_true,
            y_pred,
            os.path.join(plot_dir, "test_predictions_full.png"),
            len(y_true),
            title_prefix,
        )
        plot_scatter(y_true, y_pred, os.path.join(plot_dir, "scatter.png"), title_prefix)
        plot_error_histogram(y_true, y_pred, os.path.join(plot_dir, "error_histogram.png"), title_prefix)

    return {
        "cycle": cycle_name,
        "seq_len": seq_len,
        "category": category,
        "stimulus": aggregate_name,
        "fold": fold_idx + 1,
        "input_dim": input_dim,
        "n_train_samples": len(train_ds),
        "n_val_samples": 0 if val_ds is None else len(val_ds),
        "n_test_samples": len(test_ds),
        "best_monitor_mse": best_monitor_loss,
        "monitor_name": monitor_name,
        "test_mse": metrics["mse"],
        "test_rmse": metrics["rmse"],
        "test_mae": metrics["mae"],
        "test_r2": metrics["r2"],
        "checkpoint_path": checkpoint_path,
        "prediction_csv_path": prediction_csv_path,
        "history_csv_path": history_csv_path,
    }


def aggregate_cycle_seq_lens(args) -> List[Tuple[str, int]]:
    return [("frame", args.frame_seq_len), ("seq5", args.sequence_seq_len)]


def run_category_aggregate_training(
    category: str,
    rows: Sequence[Dict],
    args,
    device,
) -> List[Dict]:
    rows = [row for row in rows if row.get("category") == category and "x_arr" in row]
    if not rows:
        print()
        print(f"Skipping aggregate category {category}: no prepared rows.")
        return []

    input_dims = sorted(set(int(row["x_arr"].shape[1]) for row in rows))
    if len(input_dims) != 1:
        raise RuntimeError(
            f"Cannot run aggregate category {category}: incompatible feature dimensions {input_dims}."
        )

    aggregate_name = f"all_{category}_cleaned"
    fold_rows = []

    print()
    print("#" * 80)
    print(f"Starting aggregate category stage: {aggregate_name}")
    print("#" * 80)
    print("Included stimuli:")
    for row in rows:
        print(f"- {row['stimulus']} ({len(row['x_arr'])} samples, input_dim={row['x_arr'].shape[1]})")

    if args.save_plots:
        x_stack = np.vstack([row["x_arr"] for row in rows])
        y_stack = np.concatenate([row["y_arr"] for row in rows])
        save_data_overview_plots(
            x_arr=x_stack,
            y_arr=y_stack,
            out_dir=os.path.join(args.output_dir, "data_overview", aggregate_name),
        )

    for cycle_name, seq_len in aggregate_cycle_seq_lens(args):
        print()
        print("#" * 80)
        print(f"Starting aggregate cycle: {cycle_name} | seq_len={seq_len} | category: {category}")
        print("#" * 80)

        fold_indices = make_balanced_multi_kfold_samples(
            rows=rows,
            seq_len=seq_len,
            n_folds=args.n_folds,
            split=args.split,
            seed=args.seed,
        )

        for fold_idx, (train_samples, test_samples) in enumerate(fold_indices):
            set_seed(args.seed + 3000 * (fold_idx + 1) + 31 * seq_len + (0 if category == "pitch" else 700))
            fold_row = train_one_category_aggregate_fold(
                cycle_name=cycle_name,
                seq_len=seq_len,
                aggregate_name=aggregate_name,
                category=category,
                fold_idx=fold_idx,
                rows=rows,
                train_samples=train_samples,
                test_samples=test_samples,
                args=args,
                device=device,
            )
            fold_rows.append(fold_row)

    return fold_rows


def summarize_fold_rows(fold_rows: Sequence[Dict]) -> List[Dict]:
    summary_rows = []
    keys = sorted(set((row["cycle"], row["seq_len"], row["category"], row["stimulus"]) for row in fold_rows))
    metric_names = ["test_mse", "test_rmse", "test_mae", "test_r2"]
    count_names = ["n_train_samples", "n_val_samples", "n_test_samples"]

    for cycle, seq_len, category, stimulus in keys:
        rows = [row for row in fold_rows if row["cycle"] == cycle and row["stimulus"] == stimulus]
        out = {
            "cycle": cycle,
            "seq_len": seq_len,
            "category": category,
            "stimulus": stimulus,
            "input_dim": rows[0]["input_dim"] if rows else 0,
            "n_folds": len(rows),
        }

        for key in count_names:
            values = np.asarray([row[key] for row in rows], dtype=np.float64)
            out[f"{key}_mean"] = float(np.mean(values))
            out[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

        for key in metric_names:
            values = np.asarray([row[key] for row in rows], dtype=np.float64)
            out[f"{key}_mean"] = float(np.nanmean(values))
            out[f"{key}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0

        summary_rows.append(out)

    return summary_rows


def cycle_seq_lens(args) -> List[Tuple[str, int]]:
    out = []
    for cycle in args.cycles:
        if cycle == "frame":
            out.append(("frame", args.frame_seq_len))
        elif cycle == "seq5":
            out.append(("seq5", args.sequence_seq_len))
        else:
            raise RuntimeError(f"Unsupported cycle: {cycle}")
    return out


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--feature-dir", type=str, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--mocap-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--target-suffix", type=str, default="audio")
    parser.add_argument("--target-ext", type=str, default="auto", choices=["auto", "npy", ".npy", "csv", ".csv"])
    parser.add_argument("--csv-target-column", type=int, default=-1)

    parser.add_argument("--stimuli", type=str, nargs="*", default=None)
    parser.add_argument("--split", type=str, default="random", choices=["chronological", "random"])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--val-ratio-within-train", type=float, default=0.0)

    parser.add_argument("--cycles", type=str, nargs="+", default=["frame", "seq5"], choices=["frame", "seq5"])
    parser.add_argument("--frame-seq-len", type=int, default=1)
    parser.add_argument("--sequence-seq-len", type=int, default=5)

    parser.add_argument("--hand-fps", type=float, default=HAND_FPS)
    parser.add_argument("--target-fps", type=float, default=TARGET_FPS)
    parser.add_argument("--cv-fps", type=float, default=None, help="Alias for --target-fps.")
    parser.add_argument(
        "--mocap-resample-mode",
        type=str,
        default="mean",
        choices=["mean", "nearest"],
        help=(
            "How to collapse high-rate MOCAP frames to one feature row per target/CV frame. "
            "mean averages all MOCAP frames mapped to the same target frame; nearest uses the nearest MOCAP frame."
        ),
    )
    parser.add_argument("--max-nils-to-fill", type=int, default=DEFAULT_MAX_NILS_TO_FILL)

    parser.add_argument("--hand-prefixes", type=str, nargs="+", default=["right", "left", "hand"])
    parser.add_argument("--pitch-hand-max-markers", type=int, default=6)
    parser.add_argument("--volume-hand-max-markers", type=int, default=3)
    parser.add_argument("--pitch-antenna-prefix", type=str, default="pitch")
    parser.add_argument("--volume-antenna-prefix", type=str, default=None)
    parser.add_argument("--min-antenna-scale", type=float, default=1e-6)

    parser.add_argument("--range-normalization", type=str, default="minus_one_one", choices=["minus_one_one", "zero_one", "none"])
    parser.add_argument("--range-epsilon", type=float, default=1e-8)
    parser.set_defaults(use_fold_standardization=False)
    parser.add_argument("--use-fold-standardization", dest="use_fold_standardization", action="store_true")
    parser.add_argument("--no-use-fold-standardization", dest="use_fold_standardization", action="store_false")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--coord-mlp-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--scheduler-factor", type=float, default=0.75)
    parser.add_argument("--min-lr", type=float, default=1e-6)

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)

    parser.set_defaults(save_plots=True)
    parser.add_argument("--save-plots", dest="save_plots", action="store_true")
    parser.add_argument("--no-save-plots", dest="save_plots", action="store_false")
    parser.add_argument("--plot-chunk-size", type=int, default=300)

    parser.set_defaults(run_category_aggregate=True)
    parser.add_argument("--run-category-aggregate", dest="run_category_aggregate", action="store_true")
    parser.add_argument("--no-run-category-aggregate", dest="run_category_aggregate", action="store_false")

    parser.add_argument("--fold-results-csv", type=str, default=None)
    parser.add_argument("--summary-csv", type=str, default=None)

    args = parser.parse_args()

    if args.mocap_dir is None:
        args.mocap_dir = os.path.join(args.feature_dir, "mocap")

    if args.cv_fps is not None:
        args.target_fps = args.cv_fps

    if args.frame_seq_len != 1:
        print(f"Warning: frame cycle is configured with seq_len={args.frame_seq_len}, not 1.")
    if args.sequence_seq_len <= 1:
        print(f"Warning: seq5 cycle is configured with seq_len={args.sequence_seq_len}.")

    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Feature dir: {args.feature_dir}")
    print(f"MOCAP dir:   {args.mocap_dir}")

    cycles = cycle_seq_lens(args)
    print()
    print("Individual training cycles")
    for cycle_name, seq_len in cycles:
        print(f"- {cycle_name}: seq_len={seq_len}")

    print()
    print("Aggregate category cycles")
    for cycle_name, seq_len in aggregate_cycle_seq_lens(args):
        print(f"- {cycle_name}: seq_len={seq_len}")

    pairs = discover_mocap_stimuli(
        feature_dir=args.feature_dir,
        mocap_dir=args.mocap_dir,
        target_suffix=args.target_suffix,
        target_ext=args.target_ext,
    )

    if args.stimuli:
        selected = set(args.stimuli)
        pairs = [pair for pair in pairs if pair["stimulus"] in selected]
        found = set(pair["stimulus"] for pair in pairs)
        missing = selected - found
        if missing:
            raise RuntimeError(f"Requested stimuli not found: {sorted(missing)}")

    print()
    print("Discovered cleaned MOCAP/audio pairs")
    for pair in pairs:
        print(f"- {pair['stimulus']} [{pair['category']}]")
        print(f"  mocap:  {pair['mocap_path']}")
        print(f"  target: {pair['target_path']}")

    prepared = []
    for pair in pairs:
        row = load_mocap_features(
            path=pair["mocap_path"],
            stimulus=pair["stimulus"],
            args=args,
        )
        row["target_path"] = pair["target_path"]
        prepared.append(row)

    range_stats = compute_category_range_stats(prepared, epsilon=args.range_epsilon)
    write_normalization_report(
        os.path.join(args.output_dir, "mocap_category_range_normalization.csv"),
        range_stats,
    )

    for row in prepared:
        category = row["category"]
        row["x_normalized"] = apply_range_normalization(
            row["x_relative"],
            range_stats[category],
            mode=args.range_normalization,
        )

    all_fold_rows = []

    for row in prepared:
        stimulus = row["stimulus"]
        category = row["category"]

        print()
        print("#" * 80)
        print(f"Loading target and training individual stimulus: {stimulus} [{category}]")
        print("#" * 80)

        target_arr = load_target_array(row["target_path"], csv_target_column=args.csv_target_column)

        x_arr, y_arr, frames = build_fps_matched_arrays(
            feature_arr=row["x_normalized"],
            target_arr=target_arr,
            source_frames=row["frames"],
            max_nils_to_fill=args.max_nils_to_fill,
            hand_fps=args.hand_fps,
            target_fps=args.target_fps,
            mocap_resample_mode=args.mocap_resample_mode,
        )

        row["x_arr"] = x_arr
        row["y_arr"] = y_arr
        row["frames_cv"] = frames

        if args.save_plots:
            save_data_overview_plots(
                x_arr=x_arr,
                y_arr=y_arr,
                out_dir=os.path.join(args.output_dir, "data_overview", stimulus),
            )

        for cycle_name, seq_len in cycles:
            print()
            print("#" * 80)
            print(f"Starting individual cycle: {cycle_name} | seq_len={seq_len} | stimulus: {stimulus}")
            print("#" * 80)

            fold_indices = make_kfold_end_indices(
                n_frames=len(x_arr),
                seq_len=seq_len,
                n_folds=args.n_folds,
                split=args.split,
                seed=args.seed,
            )

            for fold_idx, (train_ends, test_ends) in enumerate(fold_indices):
                set_seed(args.seed + 1000 * (fold_idx + 1) + 17 * seq_len)
                fold_row = train_one_fold(
                    cycle_name=cycle_name,
                    seq_len=seq_len,
                    stimulus=stimulus,
                    category=category,
                    fold_idx=fold_idx,
                    x_arr=x_arr,
                    y_arr=y_arr,
                    frames=frames,
                    train_ends=train_ends,
                    test_ends=test_ends,
                    args=args,
                    device=device,
                )
                all_fold_rows.append(fold_row)

    if args.run_category_aggregate:
        for category in ["pitch", "volume"]:
            aggregate_rows = run_category_aggregate_training(
                category=category,
                rows=prepared,
                args=args,
                device=device,
            )
            all_fold_rows.extend(aggregate_rows)

    fold_fieldnames = [
        "cycle",
        "seq_len",
        "category",
        "stimulus",
        "fold",
        "input_dim",
        "n_train_samples",
        "n_val_samples",
        "n_test_samples",
        "best_monitor_mse",
        "monitor_name",
        "test_mse",
        "test_rmse",
        "test_mae",
        "test_r2",
        "checkpoint_path",
        "prediction_csv_path",
        "history_csv_path",
    ]

    if args.fold_results_csv is None:
        args.fold_results_csv = os.path.join(args.output_dir, "mocap_crossval_fold_results.csv")
    write_csv(args.fold_results_csv, all_fold_rows, fold_fieldnames)

    summary_rows = summarize_fold_rows(all_fold_rows)
    summary_fieldnames = list(summary_rows[0].keys()) if summary_rows else ["cycle", "seq_len", "category", "stimulus", "n_folds"]

    if args.summary_csv is None:
        args.summary_csv = os.path.join(args.output_dir, "mocap_crossval_summary.csv")
    write_csv(args.summary_csv, summary_rows, summary_fieldnames)

    if args.save_plots:
        save_summary_plots(summary_rows, args.output_dir)

    print()
    print("Final summary")
    for row in summary_rows:
        print(
            f"{row['cycle']} | seq_len={row['seq_len']} | {row['category']} | {row['stimulus']} | "
            f"MSE {row['test_mse_mean']:.6f} +/- {row['test_mse_std']:.6f} | "
            f"RMSE {row['test_rmse_mean']:.6f} +/- {row['test_rmse_std']:.6f} | "
            f"MAE {row['test_mae_mean']:.6f} +/- {row['test_mae_std']:.6f} | "
            f"R2 {row['test_r2_mean']:.6f} +/- {row['test_r2_std']:.6f}"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise
