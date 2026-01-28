import torch
from torch.utils.data import DataLoader
from data_loader import ThereminDataset
from network import ThereminRNN, ThereminMLP
from einops import rearrange, repeat


def train(model, train_dataloader, criterion, optimizer, epoch):
        model.train()   
        total_loss = 0

        for i, (audio, mocap) in enumerate(train_dataloader):

            optimizer.zero_grad()

            mocap = mocap / 5000.0

            mocap_hands = mocap[:, :27].float()
            mocap_antennas = mocap[:, 27:].float()

            output, _ = model(mocap_hands, mocap_antennas)
            #output = model(mocap_hands, mocap_antennas)

            loss = criterion(output, audio.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()


        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch}: Train Loss: {avg_loss:.6f}")






import os
import numpy as np
import matplotlib.pyplot as plt
import torch

def _to_2col_numpy(x: torch.Tensor) -> np.ndarray:
    """
    Converts a tensor shaped (..., 2) to a 2D numpy array of shape (N, 2),
    preserving temporal order as it appears in the dataloader.
    """
    x = x.detach().cpu()
    if x.ndim == 1:
        # (2,) -> (1, 2)
        x = x.unsqueeze(0)
    # flatten all leading dims, keep last dim (must be 2)
    x = x.reshape(-1, x.shape[-1])
    return x.numpy()

def _minmax_01_colwise(a: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Column-wise min-max normalization to [0, 1].
    a: (N, 2)
    """
    mn = a.min(axis=0, keepdims=True)
    mx = a.max(axis=0, keepdims=True)
    return (a - mn) / (mx - mn + eps)

def plot_gt_vs_pred_2d(gt: np.ndarray, pred: np.ndarray, plot_path: str, title: str = "GT vs Pred (normalized)"):
    """
    gt, pred: (N, 2)
    Normalizes using a shared min/max computed on the concatenation (gt+pred),
    so both are on the same 0–1 scale per column.
    """
    assert gt.shape == pred.shape and gt.shape[1] == 2, f"Expected (N,2). Got gt={gt.shape}, pred={pred.shape}"

    gt = _minmax_01_colwise(gt)
    pred = _minmax_01_colwise(pred)

    both = np.concatenate([gt, pred], axis=0)

    gt_n = both[: len(gt)]
    pred_n = both[len(gt):]

    t = np.arange(len(gt_n))

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(title)

    for d in range(2):
        ax = axes[d]
        ax.plot(t, gt_n[:, d], label=f"GT dim {d}", linewidth=1.5, alpha=0.5)      # default color
        ax.plot(t, pred_n[:, d], label=f"Pred dim {d}", linewidth=1.5, alpha=0.5)  # default color

        # Force requested colors: blue for GT, red for Pred
        ax.lines[-2].set_color("blue")
        ax.lines[-1].set_color("red")

        ax.set_ylabel(f"val {d} (0–1)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("frame")

    os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close(fig)

def eval(model, eval_dataloader, criterion, plot_path, device=None):
    model.eval()
    total_loss = 0.0

    y_true_all = []
    y_pred_all = []

    total_loss_pitch = 0.0
    total_loss_volume = 0.0

    if device is None:
        # fall back to model device
        device = next(model.parameters()).device

    with torch.no_grad():
        for i, (audio, mocap) in enumerate(eval_dataloader):
            audio = audio.to(device)
            mocap = mocap.to(device)

            mocap = mocap / 5000.0

            mocap_hands = mocap[:, :27].float()
            mocap_antennas = mocap[:, 27:].float()

            #output, _ = model(mocap_hands, mocap_antennas)
            output = model(mocap_hands, mocap_antennas)

            loss_pitch = criterion(output[0], audio[0].float())
            loss_volume = criterion(output[1], audio[1].float())
            loss = loss_pitch + loss_volume
            total_loss += float(loss.item())
            total_loss_pitch += float(loss_pitch.item())
            total_loss_volume += float(loss_volume.item())

            # collect for plotting
            y_true_all.append(audio.float())
            y_pred_all.append(output.float())

    avg_loss = total_loss / max(1, len(eval_dataloader))
    print(f"Eval Loss: {avg_loss:.6f}")
    print(f"  - Pitch Loss: {total_loss_pitch / max(1, len(eval_dataloader)):.6f}")
    print(f"  - Volume Loss: {total_loss_volume / max(1, len(eval_dataloader)):.6f}")

    # Concatenate all batches
    y_true = torch.cat(y_true_all, dim=0)
    y_pred = torch.cat(y_pred_all, dim=0)

    # Convert to (N,2) for plotting (handles (B,2) or (B,T,2), etc.)
    gt_np = _to_2col_numpy(y_true)[:2000]
    pred_np = _to_2col_numpy(y_pred)[:2000]

    gt_np = gt_np[:2000]
    pred_np = pred_np[:2000]

    diff = gt_np - pred_np
    print(diff)

    # plot the first 10 frames, using only the first 2 columns

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    plt.plot(gt_np[:,0], label='GT Pitch', color='blue')
    plt.plot(pred_np[:,0], label='Pred Pitch', color='red')
    plt.title('GT vs Pred Pitch (first 10 frames)')
    plt.xlabel('Frame')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid()
    plt.savefig('pitch_comparison.png')
    plt.close()


    # Plot GT (blue) vs Pred (red)
    plot_gt_vs_pred_2d(gt_np, pred_np, plot_path, title=f"GT vs Pred (Eval) — loss {avg_loss:.6f}")




    


if __name__ == "__main__":
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-5

    train_dataset = ThereminDataset(training=True)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = ThereminDataset(training=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    model = ThereminMLP()
    #model = ThereminRNN()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    '''for epoch in range(NUM_EPOCHS):
        train(model, train_dataloader, criterion, optimizer, epoch)
    torch.save(model.state_dict(), f"model_RNN_final.pth")'''

    # load the pth file
    model.load_state_dict(torch.load(f"model_MLP_final.pth"))

    eval_plot_path = f"eval_plot_epoch_{NUM_EPOCHS-1}.png"
    eval(model, train_dataloader, criterion, eval_plot_path) 