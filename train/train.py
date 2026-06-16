import argparse
import os
import random
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import HandSequenceDataset, chronological_split_by_frame, compute_feature_stats
from network import HandNet


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def clip_dataframes_by_max_frame(
    zed_df: pd.DataFrame,
    target_df: pd.DataFrame,
    frame_col: str,
    frame_offset: int,
    max_frame: int | None,
):

    if max_frame is None:
        return zed_df, target_df

    zed_df = zed_df.copy()
    target_df = target_df.copy()

    zed_df[frame_col] = zed_df[frame_col].astype(int)
    target_df[frame_col] = target_df[frame_col].astype(int)

    max_frame = int(max_frame)
    max_target_frame = max_frame + int(frame_offset)

    zed_df = zed_df[zed_df[frame_col] <= max_frame].copy()
    target_df = target_df[target_df[frame_col] <= max_target_frame].copy()

    if len(zed_df) == 0:
        raise RuntimeError(
            f"After applying --max-frame {max_frame}, zed_df has 0 rows."
        )

    if len(target_df) == 0:
        raise RuntimeError(
            f"After applying --max-frame {max_frame}, target_df has 0 rows. "
            f"Equivalent target max frame was {max_target_frame}."
        )

    print()
    print("Applied frame clipping:")
    print(f"Raw ZED max frame:        {max_frame}")
    print(f"Equivalent target frame:  {max_target_frame}")
    print(f"Remaining ZED rows:       {len(zed_df)}")
    print(f"Remaining target rows:    {len(target_df)}")

    return zed_df, target_df


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()

    total_loss = 0.0
    total_count = 0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()

        y_hat = model(x)
        loss = loss_fn(y_hat, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_count += x.size(0)

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_count = 0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        y_hat = model(x)
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

        y_hat = model(x).cpu().numpy()

        all_frames.append(frames)
        all_y_true.append(y)
        all_y_pred.append(y_hat)

    if len(all_frames) == 0:
        return np.array([]), np.array([]), np.array([])

    frames = np.concatenate(all_frames)
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)

    order = np.argsort(frames)

    return frames[order], y_true[order], y_pred[order]


def compute_regression_metrics(y_true, y_pred):
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


def save_prediction_csv(task_name, frames, y_true, y_pred, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)

    out_path = os.path.join(plot_dir, f"{task_name.lower()}_test_predictions.csv")

    df = pd.DataFrame(
        {
            "Frame": frames,
            "GroundTruth": y_true,
            "Prediction": y_pred,
            "Error": y_pred - y_true,
        }
    )

    df.to_csv(out_path, index=False)
    print(f"{task_name} prediction CSV saved to: {out_path}")


def plot_prediction_chunks(
    task_name,
    frames,
    y_true,
    y_pred,
    plot_dir,
    num_chunks=3,
    chunk_size=300,
):
    os.makedirs(plot_dir, exist_ok=True)

    n = len(frames)

    if n == 0:
        print(f"No predictions available for {task_name}; skipping plot.")
        return

    chunk_size = min(chunk_size, n)
    num_chunks = min(num_chunks, max(1, n // chunk_size))

    if num_chunks <= 1:
        starts = [0]
    else:
        starts = np.linspace(0, n - chunk_size, num_chunks).astype(int).tolist()

    metrics = compute_regression_metrics(y_true, y_pred)

    fig_height = 4 * len(starts)
    fig, axes = plt.subplots(
        len(starts),
        1,
        figsize=(16, fig_height),
        sharey=True,
    )

    if len(starts) == 1:
        axes = [axes]

    for ax, start in zip(axes, starts):
        end = start + chunk_size

        frame_chunk = frames[start:end]
        y_true_chunk = y_true[start:end]
        y_pred_chunk = y_pred[start:end]

        ax.plot(frame_chunk, y_true_chunk, linewidth=2.0, label="Ground truth")
        ax.plot(frame_chunk, y_pred_chunk, linewidth=2.0, linestyle="--", label="Prediction")

        ax.set_title(
            f"{task_name} test sequence | "
            f"frames {int(frame_chunk[0])} to {int(frame_chunk[-1])}"
        )
        ax.set_xlabel("Frame")
        ax.set_ylabel(task_name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    fig.suptitle(
        f"{task_name} predictions on test set\n"
        f"MSE={metrics['mse']:.6f} | "
        f"RMSE={metrics['rmse']:.6f} | "
        f"MAE={metrics['mae']:.6f} | "
        f"R2={metrics['r2']:.4f}",
        fontsize=16,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(plot_dir, f"{task_name.lower()}_test_predictions.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"{task_name} prediction plot saved to: {out_path}")


def plot_prediction_scatter(task_name, y_true, y_pred, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)

    if len(y_true) == 0:
        print(f"No predictions available for {task_name}; skipping scatter plot.")
        return

    metrics = compute_regression_metrics(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.45, s=18)

    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))

    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=2.0)

    ax.set_title(
        f"{task_name} prediction scatter\n"
        f"MSE={metrics['mse']:.6f} | RMSE={metrics['rmse']:.6f} | "
        f"MAE={metrics['mae']:.6f} | R2={metrics['r2']:.4f}"
    )
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Prediction")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    out_path = os.path.join(plot_dir, f"{task_name.lower()}_scatter.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"{task_name} scatter plot saved to: {out_path}")


def make_single_task_datasets(
    zed_train,
    zed_val,
    zed_test,
    target_df,
    hand,
    target_col,
    args,
):
    train_ds = HandSequenceDataset(
        zed_df=zed_train,
        target_df=target_df,
        seq_len=args.seq_len,
        target_col=target_col,
        hand=hand,
        frame_col=args.frame_col,
        frame_offset=args.frame_offset,
        mask=True,
    )

    val_ds = HandSequenceDataset(
        zed_df=zed_val,
        target_df=target_df,
        seq_len=args.seq_len,
        target_col=target_col,
        hand=hand,
        frame_col=args.frame_col,
        frame_offset=args.frame_offset,
        mask=False,
    )

    test_ds = HandSequenceDataset(
        zed_df=zed_test,
        target_df=target_df,
        seq_len=args.seq_len,
        target_col=target_col,
        hand=hand,
        frame_col=args.frame_col,
        frame_offset=args.frame_offset,
        mask=False,
    )

    if len(train_ds) == 0:
        raise RuntimeError(
            f"{target_col} dataset has 0 training samples.\n"
            f"Hand: {hand}\n"
            "Likely causes:\n"
            "- Frame numbers do not match between ZED and target CSV\n"
            "- Try --frame-offset -1 if ZED starts at 1 and target starts at 0\n"
            "- Try --frame-offset 0 if both start at the same frame\n"
            "- seq_len is too large\n"
            "- there are NaNs in the hand columns\n"
            "- --max-frame is too small"
        )

    if len(val_ds) == 0:
        print(f"Warning: {target_col} validation dataset has 0 samples.")

    if len(test_ds) == 0:
        print(f"Warning: {target_col} test dataset has 0 samples.")

    # Normalization happens after max-frame clipping and after chronological split.
    # Therefore x_mean/x_std are computed only from the selected training range.
    x_mean, x_std = compute_feature_stats(train_ds)

    train_ds.set_feature_stats(x_mean, x_std)
    val_ds.set_feature_stats(x_mean, x_std)
    test_ds.set_feature_stats(x_mean, x_std)

    return train_ds, val_ds, test_ds, x_mean, x_std


def train_task(
    task_name,
    hand,
    target_col,
    save_path,
    zed_train,
    zed_val,
    zed_test,
    target_df,
    args,
    device,
):
    print()
    print("=" * 80)
    print(f"Training {task_name}")
    print(f"Input hand: {hand}")
    print(f"Target:     {target_col}")
    print("=" * 80)

    train_ds, val_ds, test_ds, x_mean, x_std = make_single_task_datasets(
        zed_train=zed_train,
        zed_val=zed_val,
        zed_test=zed_test,
        target_df=target_df,
        hand=hand,
        target_col=target_col,
        args=args,
    )

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")
    print(f"Test samples:  {len(test_ds)}")

    first = train_ds[0]
    print(f"Example x shape: {first['x'].shape}")
    print(f"Example y:       {first['y']}")
    print(f"Example frame:   {first['frame']}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model = HandNet(
        input_dim=63,
        coord_mlp_dim=args.coord_mlp_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )

    loss_fn = nn.MSELoss()

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )

        val_loss = evaluate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        old_lr = get_lr(optimizer)
        scheduler.step(val_loss)
        new_lr = get_lr(optimizer)

        print(
            f"{task_name} | "
            f"Epoch {epoch:03d} | "
            f"train MSE: {train_loss:.6f} | "
            f"val MSE: {val_loss:.6f} | "
            f"lr: {new_lr:.8f}"
        )

        if new_lr < old_lr:
            print(f"{task_name} scheduler reduced LR: {old_lr:.8f} -> {new_lr:.8f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint = {
                "task_name": task_name,
                "hand": hand,
                "target_col": target_col,
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "x_mean": torch.tensor(x_mean, dtype=torch.float32),
                "x_std": torch.tensor(x_std, dtype=torch.float32),
                "best_val_loss": float(best_val_loss),
            }

            torch.save(checkpoint, save_path)

    print(f"{task_name} best val MSE: {best_val_loss:.6f}")
    print(f"Saved best model to: {save_path}")

    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss = evaluate(
        model=model,
        loader=test_loader,
        loss_fn=loss_fn,
        device=device,
    )

    frames, y_true, y_pred = collect_predictions(
        model=model,
        loader=test_loader,
        device=device,
    )

    metrics = compute_regression_metrics(y_true, y_pred)

    print(f"{task_name} test MSE:  {metrics['mse']:.6f}")
    print(f"{task_name} test RMSE: {metrics['rmse']:.6f}")
    print(f"{task_name} test MAE:  {metrics['mae']:.6f}")
    print(f"{task_name} test R2:   {metrics['r2']:.6f}")

    save_prediction_csv(
        task_name=task_name,
        frames=frames,
        y_true=y_true,
        y_pred=y_pred,
        plot_dir=args.plot_dir,
    )

    plot_prediction_chunks(
        task_name=task_name,
        frames=frames,
        y_true=y_true,
        y_pred=y_pred,
        plot_dir=args.plot_dir,
        num_chunks=args.plot_num_chunks,
        chunk_size=args.plot_chunk_size,
    )

    plot_prediction_scatter(
        task_name=task_name,
        y_true=y_true,
        y_pred=y_pred,
        plot_dir=args.plot_dir,
    )

    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--zed-csv",
        type=str,
        required=True,
        #default="/home/mmlab/Desktop/Theremin/ThereminMocap/data/features/final720_30fps_cam1.csv",
    )
    parser.add_argument(
        "--target-csv",
        type=str,
        required=True,
        #default="/home/mmlab/Desktop/Theremin/ThereminMocap/data/features/CV_final720_30fps.csv",
    )

    parser.add_argument("--frame-col", type=str, default="Frame")

    parser.add_argument("--pitch-target-col", type=str, default="Pitch_CV")
    parser.add_argument("--volume-target-col", type=str, default="Volume_CV")
    parser.add_argument("--pitch-hand", type=str, default="right", choices=["left", "right"])
    parser.add_argument("--volume-hand", type=str, default="left", choices=["left", "right"])
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--frame-offset", type=int, default=-1)

    parser.add_argument(
        "--max-frame",
        type=int,
        default=10000000000000, #3708
        help=(
            "Maximum raw ZED frame to use. "
            "The target dataframe is clipped at max_frame + frame_offset."
        ),
    )

    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--coord-mlp-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--scheduler-factor", type=float, default=0.75)
    parser.add_argument("--min-lr", type=float, default=1e-6)

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--pitch-save-path", type=str, default="checkpoints/pitch_model.pt")
    parser.add_argument("--volume-save-path", type=str, default="checkpoints/volume_model.pt")

    parser.add_argument("--plot-dir", type=str, default="plots")
    parser.add_argument("--plot-num-chunks", type=int, default=3)
    parser.add_argument("--plot-chunk-size", type=int, default=300)

    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    zed_df = pd.read_csv(args.zed_csv)
    target_df = pd.read_csv(args.target_csv)

    zed_df[args.frame_col] = zed_df[args.frame_col].astype(int)
    target_df[args.frame_col] = target_df[args.frame_col].astype(int)

    print("Original ZED frame range:")
    print(zed_df[args.frame_col].min(), zed_df[args.frame_col].max())

    print("Original target frame range:")
    print(target_df[args.frame_col].min(), target_df[args.frame_col].max())

    print("Original ZED rows:", len(zed_df))
    print("Original target rows:", len(target_df))

    zed_df, target_df = clip_dataframes_by_max_frame(
        zed_df=zed_df,
        target_df=target_df,
        frame_col=args.frame_col,
        frame_offset=args.frame_offset,
        max_frame=args.max_frame,
    )

    print()
    print("Selected ZED frame range:")
    print(zed_df[args.frame_col].min(), zed_df[args.frame_col].max())

    print("Selected target frame range:")
    print(target_df[args.frame_col].min(), target_df[args.frame_col].max())

    print("Selected ZED rows:", len(zed_df))
    print("Selected target rows:", len(target_df))

    zed_train, zed_val, zed_test = chronological_split_by_frame(
        zed_df,
        frame_col=args.frame_col,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    pitch_metrics = train_task(
        task_name="PITCH",
        hand=args.pitch_hand,
        target_col=args.pitch_target_col,
        save_path=args.pitch_save_path,
        zed_train=zed_train,
        zed_val=zed_val,
        zed_test=zed_test,
        target_df=target_df,
        args=args,
        device=device,
    )

    volume_metrics = train_task(
        task_name="VOLUME",
        hand=args.volume_hand,
        target_col=args.volume_target_col,
        save_path=args.volume_save_path,
        zed_train=zed_train,
        zed_val=zed_val,
        zed_test=zed_test,
        target_df=target_df,
        args=args,
        device=device,
    )

    print()
    print("=" * 80)
    print("Final results")
    print("=" * 80)

    print(
        f"Pitch  | "
        f"MSE: {pitch_metrics['mse']:.6f} | "
        f"RMSE: {pitch_metrics['rmse']:.6f} | "
        f"MAE: {pitch_metrics['mae']:.6f} | "
        f"R2: {pitch_metrics['r2']:.6f}"
    )

    print(
        f"Volume | "
        f"MSE: {volume_metrics['mse']:.6f} | "
        f"RMSE: {volume_metrics['rmse']:.6f} | "
        f"MAE: {volume_metrics['mae']:.6f} | "
        f"R2: {volume_metrics['r2']:.6f}"
    )

    print()
    print(f"Plots and prediction CSVs saved in: {args.plot_dir}")


if __name__ == "__main__":
    main()