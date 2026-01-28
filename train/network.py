import torch
import torch.nn as nn


class MPL_Layer(nn.Module):
    def __init__(self, input_size, output_size, activation = None, dropout = None):
        super(MPL_Layer, self).__init__()
        self.layer = nn.Linear(input_size, output_size)
        if activation:
            self.activation = nn.SiLU() if activation == 'silu' else nn.Sigmoid() if activation == 'sigmoid' else nn.Identity() 
        if dropout:
            self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.layer(x)
        if hasattr(self, 'activation'):
            x = self.activation(x)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        return x
 


class ThereminRNN(nn.Module):
    def __init__(self, hands_coords=27, antennas_coords=18, output_size=2, hidden_hands=256, hidden_antennas=128):
        super(ThereminRNN, self).__init__()

        self.MLP_hands = MPL_Layer(hands_coords, hidden_hands, "silu")
        self.MLP_antennas = MPL_Layer(antennas_coords, hidden_antennas, "silu")
        self.merge_head = MPL_Layer(hidden_hands + hidden_antennas, 128, "silu")

        self.rnn = nn.LSTM(128, 128, 2, bidirectional=False)
        self.pred_head = MPL_Layer(128, output_size)

    def forward(self, x_hand, x_ant, hidden=None):
        x1 = self.MLP_hands(x_hand)
        x2 = self.MLP_antennas(x_ant)

        x = self.merge_head(torch.cat([x1, x2], dim=1))
        out, hidden = self.rnn(x, hidden)
        out = self.pred_head(out)

        return out, hidden



class ThereminMLP(nn.Module):
    def __init__(self, hands_coords=27, antennas_coords=18, output_size=2):
        super(ThereminMLP, self).__init__()

        self.MLP_hands = MPL_Layer(hands_coords, 256, "silu", 0.1)
        self.MLP_antennas = MPL_Layer(antennas_coords, 128, "silu", 0.1)
        self.merge_head = MPL_Layer(256 + 128, 128, "silu", 0.1)
        self.pred_head = MPL_Layer(128, output_size)

    def forward(self, x_hand, x_ant):
        x1 = self.MLP_hands(x_hand)
        x2 = self.MLP_antennas(x_ant)

        x = self.merge_head(torch.cat([x1, x2], dim=1))
        out = self.pred_head(x)

        return out


if __name__ == "__main__":
    '''model = ThereminMLP()
    print(model)
    x1 = torch.randn(32, 27)
    x2 = torch.randn(32, 18)
    print(model(x1, x2))
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    model = ThereminMLPBig()
    x1 = torch.randn(32, 27)
    x2 = torch.randn(32, 18)
    print(model(x1, x2))
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    '''

    model = ThereminRNN()
    print(model)
    x1 = torch.rand(1, 27)
    x2 = torch.rand(1, 18)
    out, hidden = model(x1, x2)
    print(out.shape)