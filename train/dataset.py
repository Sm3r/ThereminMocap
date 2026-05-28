import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def get_hand_columns(df: pd.DataFrame, hand: str) -> list[str]:

    if hand not in {"left", "right"}:
        raise ValueError(f"hand must be 'left' or 'right', got {hand}")

    cols = []
    for joint_id in range(21):
        for axis in ["X", "Y", "Z"]:
            col = f"{hand}_{joint_id:02d}_{axis}"
            if col not in df.columns:
                raise ValueError(f"Missing column in mocap dataframe: {col}")
            cols.append(col)

    return cols


def chronological_split_by_frame(
    df: pd.DataFrame,
    frame_col: str = "Frame",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
):
    df = df.copy()
    df = df.sort_values(frame_col).reset_index(drop=True)

    frames = np.sort(df[frame_col].unique())

    n = len(frames)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_frames = frames[:n_train]
    val_frames = frames[n_train:n_train + n_val]
    test_frames = frames[n_train + n_val:]

    train_df = df[df[frame_col].isin(train_frames)].copy()
    val_df = df[df[frame_col].isin(val_frames)].copy()
    test_df = df[df[frame_col].isin(test_frames)].copy()

    return train_df, val_df, test_df


class HandSequenceDataset(Dataset):


    def __init__(
        self,
        zed_df: pd.DataFrame,
        target_df: pd.DataFrame,
        seq_len: int,
        target_col: str,
        hand: str,
        frame_col: str = "Frame",
        frame_offset: int = 0,
        x_mean=None,
        x_std=None,
        mask: bool = False,
        mask_activation_prob: float = 0.5,
        mask_max_steps: int = 2,
        mask_fill_value: float = 0.0,
    ):
        super().__init__()

        if hand not in {"left", "right"}:
            raise ValueError(f"hand must be 'left' or 'right', got {hand}")

        self.seq_len = seq_len
        self.target_col = target_col
        self.hand = hand
        self.frame_col = frame_col
        self.frame_offset = frame_offset

        self.mask = mask
        self.mask_activation_prob = mask_activation_prob
        self.mask_max_steps = mask_max_steps
        self.mask_fill_value = mask_fill_value

        if target_col not in target_df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found. "
                f"Available target columns: {list(target_df.columns)}"
            )

        zed_df = zed_df.copy()
        target_df = target_df.copy()

        zed_df[frame_col] = zed_df[frame_col].astype(int)
        target_df[frame_col] = target_df[frame_col].astype(int)

        zed_df[frame_col] = zed_df[frame_col] + frame_offset

        zed_df = zed_df.drop_duplicates(subset=[frame_col], keep="first")
        target_df = target_df.drop_duplicates(subset=[frame_col], keep="first")

        df = zed_df.merge(
            target_df[[frame_col, target_col]],
            on=frame_col,
            how="inner",
        )

        df = df.sort_values(frame_col).reset_index(drop=True)

        self.df = df
        self.df_by_frame = df.set_index(frame_col, drop=False)

        self.hand_cols = get_hand_columns(df, hand)

        self.x_mean = None
        self.x_std = None
        if x_mean is not None and x_std is not None:
            self.set_feature_stats(x_mean, x_std)

        self.samples = self._build_valid_samples()

    def set_feature_stats(self, x_mean, x_std):
        self.x_mean = np.asarray(x_mean, dtype=np.float32).reshape(1, -1)
        self.x_std = np.asarray(x_std, dtype=np.float32).reshape(1, -1)

        self.x_std[self.x_std < 1e-6] = 1.0

    def _build_valid_samples(self):
        samples = []

        frames = self.df[self.frame_col].to_numpy()
        available_frames = set(frames.tolist())

        for f in frames:
            start_f = f - self.seq_len + 1
            window_frames = np.arange(start_f, f + 1)

            if not set(window_frames.tolist()).issubset(available_frames):
                continue

            window = self.df_by_frame.loc[window_frames, self.hand_cols]

            if window.isna().any().any():
                continue

            target_value = self.df_by_frame.loc[f, self.target_col]

            if pd.isna(target_value):
                continue

            samples.append(int(f))

        return samples


    def _apply_step_masking(self, x: np.ndarray):
        masked_step_mask = np.zeros(self.seq_len, dtype=bool)

        if not self.mask:
            return x, masked_step_mask

        if np.random.rand() >= self.mask_activation_prob:
            return x, masked_step_mask

        max_steps = min(int(self.mask_max_steps), self.seq_len)

        if max_steps <= 0:
            return x, masked_step_mask

        if max_steps == 1:
            num_masked_steps = 1
        else:
            num_masked_steps = np.random.choice([1, 2])

        num_masked_steps = min(num_masked_steps, max_steps)

        masked_indices = np.random.choice(
            self.seq_len,
            size=num_masked_steps,
            replace=False,
        )

        x[masked_indices, :] = self.mask_fill_value
        masked_step_mask[masked_indices] = True

        return x, masked_step_mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f = self.samples[idx]

        start_f = f - self.seq_len + 1
        window_frames = np.arange(start_f, f + 1)

        x = self.df_by_frame.loc[window_frames, self.hand_cols].to_numpy(dtype=np.float32)
        if self.x_mean is not None and self.x_std is not None:
            x = (x - self.x_mean) / self.x_std

        x, masked_step_mask = self._apply_step_masking(x)

        y = np.float32(self.df_by_frame.loc[f, self.target_col])

        return {
            "x": torch.from_numpy(x),
            "y": torch.tensor(y, dtype=torch.float32),
            "frame": torch.tensor(f, dtype=torch.long),
            "masked_step_mask": torch.from_numpy(masked_step_mask),
        }


@torch.no_grad()
def compute_feature_stats(dataset: Dataset, batch_size: int = 512):

    old_mask_value = None

    if hasattr(dataset, "mask"):
        old_mask_value = dataset.mask
        dataset.mask = False

    try:
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

    finally:
        if old_mask_value is not None:
            dataset.mask = old_mask_value