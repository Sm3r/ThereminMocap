#!/usr/bin/env python3
"""
Plot target vs prediction chunks for aggregated ThereminMocap models.

This script assumes you trained aggregated category models using the previous
aggregated pitch/volume cross-validation script.

For each category:

    pitch  -> uses the free_pitch dataset by default
    volume -> uses the free_volume dataset by default

It loads one checkpoint per category, runs inference on the corresponding free
stimulus, selects 10 random chunks of 100 contiguous prediction samples, and
saves target+prediction plots.

Default checkpoint layout expected:

    <aggregated-run-dir>/pitch/<cycle>/fold_01/checkpoints/best_model.pt
    <aggregated-run-dir>/volume/<cycle>/fold_01/checkpoints/best_model.pt

Example:

    python plot_free_chunks_aggregated.py \
        --aggregated-run-dir /home/mmlab/Desktop/Theremin/ThereminMocap/runs_aggregated_pitch_volume_cv \
        --feature-dir /home/mmlab/Desktop/Theremin/ThereminMocap/data/features \
        --cycle seq5 \
        --fold 1

This will produce:

    <output-dir>/pitch/chunk_01.png ... chunk_10.png
    <output-dir>/volume/chunk_01.png ... chunk_10.png
"""

import argparse
import csv
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset

from network import HandNet


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_FEATURE_DIR = "/home/mmlab/Desktop/Theremin/ThereminMocap/data/features"
DEFAULT_AGGREGATED_RUN_DIR = (
    "/home/mmlab/Desktop/Theremin/ThereminMocap/runs_aggregated_pitch_volume_cv"
)
DEFAULT_OUTPUT_DIR = (
    "/home/mmlab/Desktop/Theremin/ThereminMocap/free_chunk_predictions_aggregated"
)

HAND_FPS = 30.0
TARGET_FPS = 60.0
MAX_NILS_TO_FILL = 10


# =============================================================================
# Reproducibility / loading
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        raise RuntimeError(
            f"Could not load NumPy file: {path}\nOriginal error: {exc}"
        ) from exc

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


def load_array(path: str, csv_target_column: int) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        return load_npy_array(path)

    if ext == ".csv":
        return load_csv_array(path, csv_target_column=csv_target_column)

    raise RuntimeError(
        f"Unsupported file extension for {path}. Expected .npy or .csv."
    )


# =============================================================================
# FPS matching / cleaning
# =============================================================================

def clean_target_array(target_arr: np.ndarray) -> np.ndarray:
    if target_arr.ndim == 2 and target_arr.shape[1] == 1:
        target_arr = target_arr[:, 0]

    if target_arr.ndim != 1:
        raise RuntimeError(
            f"Expected target array [frames] or [frames, 1], got {target_arr.shape}"
        )

    return target_arr.astype(np.float32)


def clean_hand_array(
    hand_arr: np.ndarray,
    max_nils_to_fill: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if hand_arr.ndim != 2:
        raise RuntimeError(f"Expected hand array [frames, 63], got {hand_arr.shape}")

    if hand_arr.shape[1] != 63:
        raise RuntimeError(f"Expected 63 hand features, got {hand_arr.shape[1]}")

    nil_count = np.isnan(hand_arr).sum(axis=1)
    keep_mask = nil_count <= max_nils_to_fill

    cleaned = hand_arr[keep_mask].copy()
    cleaned[np.isnan(cleaned)] = 0.0

    return cleaned.astype(np.float32), keep_mask


def build_fps_matched_arrays(
    hand_arr: np.ndarray,
    target_arr: np.ndarray,
    max_nils_to_fill: int,
    hand_fps: float,
    target_fps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if hand_fps <= 0:
        raise RuntimeError(f"hand_fps must be > 0, got {hand_fps}")

    if target_fps <= 0:
        raise RuntimeError(f"target_fps must be > 0, got {target_fps}")

    raw_hand_len = len(hand_arr)
    raw_target_len = len(target_arr)

    if raw_hand_len == 0:
        raise RuntimeError("Hand array has 0 frames.")

    if raw_target_len == 0:
        raise RuntimeError("Target array has 0 frames.")

    if hand_arr.ndim != 2:
        raise RuntimeError(f"Expected hand array [frames, 63], got {hand_arr.shape}")

    if hand_arr.shape[1] != 63:
        raise RuntimeError(f"Expected 63 hand features, got {hand_arr.shape[1]}")

    target_arr = clean_target_array(target_arr)

    hand_frame_idx = np.arange(raw_hand_len, dtype=np.float64)
    hand_time_s = hand_frame_idx / float(hand_fps)

    target_idx = np.rint(hand_time_s * float(target_fps)).astype(np.int64)
    target_idx = np.clip(target_idx, 0, raw_target_len - 1)

    y_for_hand = target_arr[target_idx]

    x_clean, keep_mask = clean_hand_array(hand_arr, max_nils_to_fill)
    y_clean = y_for_hand[keep_mask]
    frames_clean = np.arange(raw_hand_len, dtype=np.int64)[keep_mask]

    valid_y = np.isfinite(y_clean)
    x_clean = x_clean[valid_y]
    y_clean = y_clean[valid_y]
    frames_clean = frames_clean[valid_y]

    if len(x_clean) == 0:
        raise RuntimeError("Final arrays have 0 rows after cleaning.")

    return (
        x_clean.astype(np.float32),
        y_clean.astype(np.float32),
        frames_clean.astype(np.int64),
    )


# =============================================================================
# Dataset
# =============================================================================

class FreeSequenceDataset(Dataset):
    def __init__(
        self,
        x_arr: np.ndarray,
        y_arr: np.ndarray,
        frames: np.ndarray,
        seq_len: int,
        x_mean: np.ndarray,
        x_std: np.ndarray,
    ):
        super().__init__()

        self.x = np.asarray(x_arr, dtype=np.float32)
        self.y = np.asarray(y_arr, dtype=np.float32).reshape(-1)
        self.frames = np.asarray(frames, dtype=np.int64).reshape(-1)
        self.seq_len = int(seq_len)

        if self.x.ndim != 2:
            raise RuntimeError(f"Expected x_arr [frames, features], got {self.x.shape}")

        if len(self.x) != len(self.y):
            raise RuntimeError(f"x/y length mismatch: {len(self.x)} vs {len(self.y)}")

        if len(self.x) != len(self.frames):
            raise RuntimeError(
                f"x/frames length mismatch: {len(self.x)} vs {len(self.frames)}"
            )

        if self.seq_len <= 0:
            raise RuntimeError(f"seq_len must be > 0, got {self.seq_len}")

        if len(self.x) < self.seq_len:
            raise RuntimeError(
                f"Not enough frames for seq_len={self.seq_len}: got {len(self.x)}"
            )

        self.end_indices = np.arange(self.seq_len - 1, len(self.x), dtype=np.int64)

        self.x_mean = np.asarray(x_mean, dtype=np.float32).reshape(1, -1)
        self.x_std = np.asarray(x_std, dtype=np.float32).reshape(1, -1)
        self.x_std[self.x_std < 1e-6] = 1.0

    def __len__(self) -> int:
        return len(self.end_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        end = int(self.end_indices[idx])
        start = end - self.seq_len + 1

        x = self.x[start:end + 1].copy()
        x = (x - self.x_mean) / self.x_std

        y = np.float32(self.y[end])
        frame = int(self.frames[end])
        sample_index = int(idx)

        return {
            "x": torch.from_numpy(x),
            "y": torch.tensor(y, dtype=torch.float32),
            "frame": torch.tensor(frame, dtype=torch.long),
            "sample_index": torch.tensor(sample_index, dtype=torch.long),
        }


# =============================================================================
# Model / inference
# =============================================================================

def checkpoint_arg(
    checkpoint: Dict,
    name: str,
    fallback,
):
    args = checkpoint.get("args", {})

    if isinstance(args, dict) and name in args:
        return args[name]

    return fallback


def make_model_from_checkpoint(
    checkpoint: Dict,
    device,
    fallback_args,
):
    coord_mlp_dim = checkpoint_arg(
        checkpoint,
        "coord_mlp_dim",
        fallback_args.coord_mlp_dim,
    )
    hidden_dim = checkpoint_arg(
        checkpoint,
        "hidden_dim",
        fallback_args.hidden_dim,
    )
    num_layers = checkpoint_arg(
        checkpoint,
        "num_layers",
        fallback_args.num_layers,
    )
    dropout = checkpoint_arg(
        checkpoint,
        "dropout",
        fallback_args.dropout,
    )

    model = HandNet(
        input_dim=63,
        coord_mlp_dim=coord_mlp_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def tensor_or_array_to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    return np.asarray(value)


@torch.no_grad()
def collect_predictions(
    model,
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    all_sample_indices = []
    all_frames = []
    all_y_true = []
    all_y_pred = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        y_hat = model(x).reshape_as(y)

        all_sample_indices.append(batch["sample_index"].cpu().numpy())
        all_frames.append(batch["frame"].cpu().numpy())
        all_y_true.append(y.cpu().numpy())
        all_y_pred.append(y_hat.cpu().numpy())

    if not all_sample_indices:
        return np.array([]), np.array([]), np.array([]), np.array([])

    sample_indices = np.concatenate(all_sample_indices)
    frames = np.concatenate(all_frames)
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)

    order = np.argsort(sample_indices)

    return (
        sample_indices[order],
        frames[order],
        y_true[order],
        y_pred[order],
    )


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    if len(y_true) == 0:
        return {
            "mse": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
        }

    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(math.sqrt(mse))

    denom = np.sum((y_true - np.mean(y_true)) ** 2)

    if denom < 1e-12:
        r2 = float("nan")
    else:
        r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


# =============================================================================
# Chunk selection / plotting
# =============================================================================

def choose_random_chunks(
    n_samples: int,
    chunk_size: int,
    n_chunks: int,
    seed: int,
    allow_overlap: bool,
) -> List[Tuple[int, int]]:
    if n_samples < chunk_size:
        raise RuntimeError(
            f"Cannot select chunks of length {chunk_size}: only {n_samples} samples."
        )

    rng = np.random.default_rng(seed)

    if allow_overlap:
        starts = rng.integers(
            low=0,
            high=n_samples - chunk_size + 1,
            size=n_chunks,
        )
        return [(int(s), int(s + chunk_size)) for s in starts]

    possible_starts = list(range(0, n_samples - chunk_size + 1))
    rng.shuffle(possible_starts)

    selected = []

    for start in possible_starts:
        stop = start + chunk_size

        overlaps = False
        for existing_start, existing_stop in selected:
            if start < existing_stop and stop > existing_start:
                overlaps = True
                break

        if not overlaps:
            selected.append((start, stop))

        if len(selected) >= n_chunks:
            break

    if len(selected) < n_chunks:
        print(
            f"Warning: only found {len(selected)} non-overlapping chunks. "
            f"Falling back to overlapping chunks for the rest."
        )

        while len(selected) < n_chunks:
            start = int(rng.integers(0, n_samples - chunk_size + 1))
            selected.append((start, start + chunk_size))

    selected = sorted(selected, key=lambda x: x[0])

    return selected


def save_prediction_csv(
    frames: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    data = np.column_stack(
        [
            frames,
            y_true,
            y_pred,
            y_pred - y_true,
        ]
    )

    np.savetxt(
        out_path,
        data,
        delimiter=",",
        header="Frame,GroundTruth,Prediction,Error",
        comments="",
        fmt=["%d", "%.8f", "%.8f", "%.8f"],
    )


def plot_chunk(
    category: str,
    stimulus: str,
    cycle: str,
    seq_len: int,
    chunk_id: int,
    start: int,
    stop: int,
    frames: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
    target_color: str = "#326ebc",
    prediction_color: str = "#f161cd",
    target_linewidth: float = 3.5,
    prediction_linewidth: float = 2.5,
    axis_label_fontsize: int = 18,
    tick_labelsize: int = 18,
    legend_fontsize: int = 20,
    grid_alpha: float = 0.32,
    grid_linewidth: float = 0.9,
) -> Dict:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    chunk_frames = frames[start:stop]
    chunk_true = y_true[start:stop]
    chunk_pred = y_pred[start:stop]

    metrics = compute_regression_metrics(chunk_true, chunk_pred)

    fig, ax = plt.subplots(figsize=(14, 3.6))

    ax.plot(
        chunk_frames,
        chunk_true,
        color=target_color,
        linewidth=target_linewidth,
        solid_capstyle="round",
        label="Target",
    )

    ax.plot(
        chunk_frames,
        chunk_pred,
        color=prediction_color,
        linewidth=prediction_linewidth,
        linestyle=":",
        dash_capstyle="round",
        label="Prediction",
    )

    ax.set_xlabel("Frame", fontsize=axis_label_fontsize, labelpad=10)
    ax.set_ylabel("Target", fontsize=axis_label_fontsize, labelpad=10)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_labelsize,
        width=1.4,
        length=6,
    )

    ax.grid(
        True,
        which="major",
        alpha=grid_alpha,
        linewidth=grid_linewidth,
    )

    ax.legend(
        loc="best",
        fontsize=legend_fontsize,
        frameon=True,
        framealpha=0.92,
        borderpad=0.55,
        handlelength=2.6,
        handletextpad=0.7,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.4)

    ax.margins(x=0.01, y=0.12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "category": category,
        "stimulus": stimulus,
        "cycle": cycle,
        "seq_len": seq_len,
        "chunk": chunk_id,
        "start_sample": start,
        "stop_sample": stop,
        "start_frame": int(chunk_frames[0]),
        "stop_frame": int(chunk_frames[-1]),
        "mse": metrics["mse"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "plot_path": out_path,
    }

def write_csv(
    path: str,
    rows: Sequence[Dict],
    fieldnames: Sequence[str],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


# =============================================================================
# File resolution
# =============================================================================

def resolve_checkpoint_path(
    category: str,
    explicit_path: Optional[str],
    aggregated_run_dir: str,
    cycle: str,
    fold: int,
) -> str:
    if explicit_path:
        path = explicit_path
    else:
        path = os.path.join(
            aggregated_run_dir,
            category,
            cycle,
            f"fold_{fold:02d}",
            "checkpoints",
            "best_model.pt",
        )

    if not os.path.exists(path):
        raise RuntimeError(
            f"Checkpoint not found for category={category}:\n"
            f"  {path}\n"
            "Pass an explicit checkpoint with --pitch-checkpoint or "
            "--volume-checkpoint if your layout differs."
        )

    return path


def resolve_free_paths(
    feature_dir: str,
    stimulus: str,
    target_suffix: str,
    target_ext: str,
) -> Tuple[str, str]:
    target_ext = target_ext if target_ext.startswith(".") else f".{target_ext}"

    hand_path = os.path.join(feature_dir, f"{stimulus}_hand.npy")
    target_path = os.path.join(feature_dir, f"{stimulus}_{target_suffix}{target_ext}")

    if not os.path.exists(hand_path):
        raise RuntimeError(f"Missing hand file: {hand_path}")

    if not os.path.exists(target_path):
        raise RuntimeError(f"Missing target file: {target_path}")

    return hand_path, target_path


# =============================================================================
# Category runner
# =============================================================================

def run_category(
    category: str,
    free_stimulus: str,
    checkpoint_path: str,
    args,
    device,
) -> List[Dict]:
    print()
    print("=" * 80)
    print(f"Category: {category}")
    print(f"Free stimulus: {free_stimulus}")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 80)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    seq_len = int(checkpoint.get("seq_len", args.seq_len_override or 1))
    cycle = str(checkpoint.get("cycle", args.cycle))

    x_mean = tensor_or_array_to_numpy(checkpoint["x_mean"]).astype(np.float32)
    x_std = tensor_or_array_to_numpy(checkpoint["x_std"]).astype(np.float32)

    model = make_model_from_checkpoint(
        checkpoint=checkpoint,
        device=device,
        fallback_args=args,
    )

    hand_path, target_path = resolve_free_paths(
        feature_dir=args.feature_dir,
        stimulus=free_stimulus,
        target_suffix=args.target_suffix,
        target_ext=args.target_ext,
    )

    print(f"Hand file:   {hand_path}")
    print(f"Target file: {target_path}")
    print(f"seq_len:     {seq_len}")
    print(f"cycle:       {cycle}")

    hand_arr = load_array(hand_path, csv_target_column=args.csv_target_column)
    target_arr = load_array(target_path, csv_target_column=args.csv_target_column)

    x_arr, y_arr, frames = build_fps_matched_arrays(
        hand_arr=hand_arr,
        target_arr=target_arr,
        max_nils_to_fill=args.max_nils_to_fill,
        hand_fps=args.hand_fps,
        target_fps=args.target_fps,
    )

    dataset = FreeSequenceDataset(
        x_arr=x_arr,
        y_arr=y_arr,
        frames=frames,
        seq_len=seq_len,
        x_mean=x_mean,
        x_std=x_std,
    )

    sample_indices, pred_frames, y_true, y_pred = collect_predictions(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    overall = compute_regression_metrics(y_true, y_pred)

    print()
    print(f"Free-set prediction samples: {len(y_true)}")
    print(f"Overall MSE:  {overall['mse']:.8f}")
    print(f"Overall MAE:  {overall['mae']:.8f}")
    print(f"Overall RMSE: {overall['rmse']:.8f}")
    print(f"Overall R2:   {overall['r2']:.6f}")

    category_out_dir = os.path.join(args.output_dir, category)
    os.makedirs(category_out_dir, exist_ok=True)

    save_prediction_csv(
        frames=pred_frames,
        y_true=y_true,
        y_pred=y_pred,
        out_path=os.path.join(category_out_dir, "full_predictions.csv"),
    )

    chunks = choose_random_chunks(
        n_samples=len(y_true),
        chunk_size=args.chunk_size,
        n_chunks=args.n_chunks,
        seed=args.seed + (0 if category == "pitch" else 10000),
        allow_overlap=args.allow_overlap,
    )

    rows = []

    for chunk_id, (start, stop) in enumerate(chunks, start=1):
        plot_path = os.path.join(category_out_dir, f"chunk_{chunk_id:02d}.png")

        row = plot_chunk(
            category=category,
            stimulus=free_stimulus,
            cycle=cycle,
            seq_len=seq_len,
            chunk_id=chunk_id,
            start=start,
            stop=stop,
            frames=pred_frames,
            y_true=y_true,
            y_pred=y_pred,
            out_path=plot_path,
        )

        rows.append(row)

        print(f"Saved {category} chunk {chunk_id:02d}: {plot_path}")

    summary_path = os.path.join(category_out_dir, "chunk_summary.csv")
    write_csv(
        summary_path,
        rows,
        fieldnames=[
            "category",
            "stimulus",
            "cycle",
            "seq_len",
            "chunk",
            "start_sample",
            "stop_sample",
            "start_frame",
            "stop_frame",
            "mse",
            "mae",
            "rmse",
            "r2",
            "plot_path",
        ],
    )

    print(f"Chunk summary saved to: {summary_path}")

    return rows


# =============================================================================
# CLI / main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--feature-dir", type=str, default=DEFAULT_FEATURE_DIR)
    parser.add_argument(
        "--aggregated-run-dir",
        type=str,
        default=DEFAULT_AGGREGATED_RUN_DIR,
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--target-suffix", type=str, default="audio")
    parser.add_argument(
        "--target-ext",
        type=str,
        default="npy",
        choices=["npy", ".npy", "csv", ".csv"],
    )
    parser.add_argument("--csv-target-column", type=int, default=-1)

    parser.add_argument("--free-pitch-stimulus", type=str, default="free_pitch")
    parser.add_argument("--free-volume-stimulus", type=str, default="free_volume")

    parser.add_argument(
        "--cycle",
        type=str,
        default="seq5",
        choices=["frame", "seq5"],
        help="Used only for automatic checkpoint path resolution.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Used only for automatic checkpoint path resolution.",
    )

    parser.add_argument("--pitch-checkpoint", type=str, default=None)
    parser.add_argument("--volume-checkpoint", type=str, default=None)

    parser.add_argument("--seq-len-override", type=int, default=None)

    parser.add_argument("--hand-fps", type=float, default=HAND_FPS)
    parser.add_argument("--target-fps", type=float, default=TARGET_FPS)
    parser.add_argument("--cv-fps", type=float, default=None)
    parser.add_argument("--max-nils-to-fill", type=int, default=MAX_NILS_TO_FILL)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--n-chunks", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--allow-overlap", action="store_true")

    parser.add_argument("--seed", type=int, default=1234)

    # Fallback architecture values, used only if absent from checkpoint["args"].
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--coord-mlp-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)

    args = parser.parse_args()

    if args.cv_fps is not None:
        args.target_fps = args.cv_fps

    if args.n_chunks <= 0:
        raise RuntimeError("--n-chunks must be > 0")

    if args.chunk_size <= 0:
        raise RuntimeError("--chunk-size must be > 0")

    if args.fold <= 0:
        raise RuntimeError("--fold must be >= 1")

    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    pitch_checkpoint = resolve_checkpoint_path(
        category="pitch",
        explicit_path=args.pitch_checkpoint,
        aggregated_run_dir=args.aggregated_run_dir,
        cycle=args.cycle,
        fold=args.fold,
    )

    volume_checkpoint = resolve_checkpoint_path(
        category="volume",
        explicit_path=args.volume_checkpoint,
        aggregated_run_dir=args.aggregated_run_dir,
        cycle=args.cycle,
        fold=args.fold,
    )

    all_rows = []

    all_rows.extend(
        run_category(
            category="pitch",
            free_stimulus=args.free_pitch_stimulus,
            checkpoint_path=pitch_checkpoint,
            args=args,
            device=device,
        )
    )

    all_rows.extend(
        run_category(
            category="volume",
            free_stimulus=args.free_volume_stimulus,
            checkpoint_path=volume_checkpoint,
            args=args,
            device=device,
        )
    )

    combined_summary_path = os.path.join(args.output_dir, "combined_chunk_summary.csv")

    write_csv(
        combined_summary_path,
        all_rows,
        fieldnames=[
            "category",
            "stimulus",
            "cycle",
            "seq_len",
            "chunk",
            "start_sample",
            "stop_sample",
            "start_frame",
            "stop_frame",
            "mse",
            "mae",
            "rmse",
            "r2",
            "plot_path",
        ],
    )

    print()
    print("Done.")
    print(f"Pitch plots:  {os.path.join(args.output_dir, 'pitch')}")
    print(f"Volume plots: {os.path.join(args.output_dir, 'volume')}")
    print(f"Summary CSV:  {combined_summary_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)