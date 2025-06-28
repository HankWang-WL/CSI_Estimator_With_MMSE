import torch
import torch.nn as nn

# ===== 3D CNN baseline =====
class SimpleCSINet3D(nn.Module):
    """
    簡單的 3D CNN 模型：輸入 [batch, 4, num_rx, num_tx, pilot_length]
        4 = x_real, x_imag, y_real, y_imag
    輸出 [batch, num_rx, num_tx, pilot_length, 2] (2 = h_real, h_imag)
    """
    def __init__(self, num_rx=4, num_tx=4, pilot_length=8, out_dim2=2, dropout_p=0.2):
        super(SimpleCSINet3D, self).__init__()
        self.num_rx = num_rx
        self.num_tx = num_tx
        self.pilot_length = pilot_length
        self.out_dim2 = out_dim2

        self.cnn = nn.Sequential(
            nn.Conv3d(in_channels=4, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 最後一層 1x1x1 卷積將通道降到 2（實部與虛部）
        self.final_conv = nn.Conv3d(in_channels=8, out_channels=out_dim2, kernel_size=1)

    def forward(self, x):
        x = self.cnn(x)  # [batch, 8, num_rx, num_tx, pilot_length]
        x = self.final_conv(x)  # [batch, 2, num_rx, num_tx, pilot_length]
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # [batch, num_rx, num_tx, pilot_length, 2]
        return x


# ===== LSTM baseline =====
class LSTMCSINet(nn.Module):
    """
    LSTM baseline：每個 pilot 長度作為 time step，輸入為 flattened channel features
    輸入 [batch, 4, num_rx, num_tx, pilot_length]
    輸出 [batch, num_rx, num_tx, pilot_length, 2]
    """
    def __init__(self, num_rx=4, num_tx=4, pilot_length=8, hidden_size=128, num_layers=2, out_dim2=2, dropout_p=0.2):
        super(LSTMCSINet, self).__init__()
        self.num_rx = num_rx
        self.num_tx = num_tx
        self.pilot_length = pilot_length
        self.out_dim2 = out_dim2
        self.input_dim = 4 * num_rx * num_tx

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_p if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_rx * num_tx * out_dim2)

    def forward(self, x):
        B = x.shape[0]
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # [B, pilot_length, 4, rx, tx]
        x = x.view(B, self.pilot_length, -1)  # [B, pilot_length, input_dim]
        out, _ = self.lstm(x)
        out = self.fc(out)  # [B, pilot_length, rx*tx*2]
        out = out.view(B, self.pilot_length, self.num_rx, self.num_tx, self.out_dim2)
        out = out.permute(0, 2, 3, 1, 4).contiguous()
        return out


# ===== Transformer baseline =====
class TransformerCSINet(nn.Module):
    """
    Transformer baseline：將每個 pilot 視為序列單位
    輸入 [batch, 4, num_rx, num_tx, pilot_length]
    輸出 [batch, num_rx, num_tx, pilot_length, 2]
    """
    def __init__(self, num_rx=4, num_tx=4, pilot_length=8, d_model=128, nhead=4, num_layers=2, out_dim2=2, dropout_p=0.2):
        super(TransformerCSINet, self).__init__()
        self.num_rx = num_rx
        self.num_tx = num_tx
        self.pilot_length = pilot_length
        self.out_dim2 = out_dim2
        self.input_dim = 4 * num_rx * num_tx

        self.embedding = nn.Linear(self.input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout_p,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_rx * num_tx * out_dim2)

    def forward(self, x):
        B = x.shape[0]
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # [B, pilot_length, 4, rx, tx]
        x = x.view(B, self.pilot_length, -1)  # [B, pilot_length, input_dim]
        x = self.embedding(x)  # [B, pilot_length, d_model]
        out = self.transformer(x)  # [B, pilot_length, d_model]
        out = self.fc(out)  # [B, pilot_length, rx * tx * 2]
        out = out.view(B, self.pilot_length, self.num_rx, self.num_tx, self.out_dim2)
        out = out.permute(0, 2, 3, 1, 4).contiguous()
        return out
