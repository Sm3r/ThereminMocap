import torch
import torch.nn as nn


class HandNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 63,
        coord_mlp_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.coord_encoder = nn.Sequential(
            nn.Linear(input_dim, coord_mlp_dim),
            nn.SiLU(),
            nn.LayerNorm(coord_mlp_dim),
            nn.Dropout(dropout),
            nn.Linear(coord_mlp_dim, coord_mlp_dim),
            nn.SiLU(),
            nn.LayerNorm(coord_mlp_dim),
        )

        self.lstm = nn.LSTM(
            input_size=coord_mlp_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        z = self.coord_encoder(x)

        lstm_out, _ = self.lstm(z)
        last = lstm_out[:, -1, :]

        y_hat = self.regressor(last).squeeze(-1)

        return y_hat


if __name__ == "__main__":
    model = HandNet()
    print(f"Parameters number: {sum(p.numel() for p in model.parameters())}")
    x = torch.randn(4, 16, 63)
    y_hat = model(x)
    print(y_hat.shape)

    import time

    start = time.time()
    for _ in range(100):
        y_hat = model(x)
    end = time.time()

    print(f"100 iters: {end - start:.2f} seconds")
