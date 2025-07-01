import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

from model import SimpleCSINet3D, LSTMCSINet, TransformerCSINet
from dataset import CSIDataset
from mmse_baseline import mmse_estimation
from config import config

# ========== 1. 資料集設定 ==========
USE_DEEPMIMO = config["use_deepmimo"]
num_tx = config["num_tx"]
num_rx = config["num_rx"]
pilot_length = config["pilot_length"]
snr_db = config["snr_db"]
quant_bits = config["quant_bits"]
model_type = config["model_type"]

if USE_DEEPMIMO:
    data_path = config["data_path"]
    dataset = CSIDataset(
        data_path=data_path,
        num_rx=num_rx,
        num_tx=num_tx,
        pilot_length=pilot_length,
        quant_bits=quant_bits,
        snr_db=snr_db
    )
else:
    dataset = CSIDataset(
        num_samples=config["num_samples"],
        pilot_length=pilot_length,
        snr_db=snr_db,
        num_tx=num_tx,
        exp_decay=config["exp_decay"],
        quant_bits=quant_bits
    )

print("資料集長度：", len(dataset))

# ========== 2. 切分訓練/驗證集 ==========
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

# ========== 3. 建立模型、Loss、Optimizer ==========
if model_type == "cnn":
    model = SimpleCSINet3D(num_rx=num_rx, num_tx=num_tx, pilot_length=pilot_length, out_dim2=2)
elif model_type == "lstm":
    model = LSTMCSINet(num_rx=num_rx, num_tx=num_tx, pilot_length=pilot_length, out_dim2=2)
elif model_type == "transformer":
    model = TransformerCSINet(num_rx=num_rx, num_tx=num_tx, pilot_length=pilot_length, out_dim2=2)
else:
    raise ValueError(f"Unknown model_type: {model_type}")

optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
criterion = nn.MSELoss()

train_losses = []
val_losses = []

# ========== 4. 開始訓練 ==========
for epoch in range(config["train_epoch"]):
    total_loss = 0
    model.train()
    for x, y, h, scale in train_loader:
        input_xy = torch.cat([x, y], dim=-1)
        input_cnn = input_xy.permute(0, 4, 1, 2, 3).contiguous()

        pred = model(input_cnn)
        loss = criterion(pred, h)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # 驗證
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y, h, scale in val_loader:
            input_xy = torch.cat([x, y], dim=-1)
            input_cnn = input_xy.permute(0, 4, 1, 2, 3).contiguous()
            pred = model(input_cnn)
            loss = criterion(pred, h)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch + 1}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

# ========== 5. MMSE Baseline（正確版本） ==========
mmse_losses = []
model.eval()
with torch.no_grad():
    for x, y, h, scale in val_loader:
        h_mmse = mmse_estimation(x, y)

        # 還原 channel 大小
        view_shape = [h.shape[0]] + [1] * (h.ndim - 1)

        loss = criterion(h_mmse, h)  
        mmse_losses.append(loss.item())  

# ✅ 真正的平均 loss
mse = sum(mmse_losses) / len(mmse_losses)
print(f"\n MMSE Baseline MSE（正確平均）：{mse:.4f}\n")


# ========== 6. 畫圖 ==========
x, y, h, scale = next(iter(val_loader))
input_xy = torch.cat([x, y], dim=-1)
input_cnn = input_xy.permute(0, 4, 1, 2, 3).contiguous()

with torch.no_grad():
    pred_h = model(input_cnn)

true_h = h[0].numpy().reshape(num_rx, num_tx, pilot_length, 2)
pred_h = pred_h[0].cpu().numpy().reshape(num_rx, num_tx, pilot_length, 2)

# ========== 建立儲存資料夾 ==========
save_dir = "results"
os.makedirs(save_dir, exist_ok=True)

# ========== 根據模型與資料類型設定前綴 ==========
prefix = f"{'DeepMIMO' if USE_DEEPMIMO else 'Rayleigh'}_{model_type}"

# ========== 圖 1: loss 曲線圖 ==========
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Train Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, marker='s', label='Validation Loss')
plt.hlines(mse, 1, len(train_losses), colors='r', linestyles='dashed', label='MMSE Baseline')
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Performance: Train vs Validation vs MMSE Baseline")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_dir}/{prefix}_LossCurve.png")

# ========== 圖 2: 殘差分布圖==========
plt.figure(figsize=(7, 4))
residual = pred_h - true_h
plt.hist(residual.flatten(), bins=80, color='steelblue', edgecolor='black', alpha=0.75)
plt.title("Residual Distribution (Predicted - True)", fontsize=14)
plt.xlabel("Prediction Error", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.savefig(f"{save_dir}/{prefix}_ResidualHistogram.png")



# ========== 圖 3: Heatmap ==========
plt.figure(figsize=(12, 5))

# Real part heatmap - True
plt.subplot(2, 2, 1)
real_true = true_h[:, :, :, 0].mean(axis=2)
plt.imshow(real_true, cmap='viridis', aspect='auto')
plt.xticks(ticks=range(num_tx), labels=range(num_tx))
plt.yticks(ticks=range(num_rx), labels=range(num_rx))
plt.colorbar()
plt.title("True H Real (avg over pilot)")
plt.xlabel("TX antenna")
plt.ylabel("RX antenna")

# Real part heatmap - Predicted
plt.subplot(2, 2, 2)
real_pred = pred_h[:, :, :, 0].mean(axis=2)
plt.imshow(real_pred, cmap='viridis', aspect='auto')
plt.xticks(ticks=range(num_tx), labels=range(num_tx))
plt.yticks(ticks=range(num_rx), labels=range(num_rx))
plt.colorbar()
plt.title("Predicted H Real (avg over pilot)")
plt.xlabel("TX antenna")
plt.ylabel("RX antenna")

# Imag part heatmap - True
plt.subplot(2, 2, 3)
imag_true = true_h[:, :, :, 1].mean(axis=2)
plt.imshow(imag_true, cmap='viridis', aspect='auto')
plt.xticks(ticks=range(num_tx), labels=range(num_tx))
plt.yticks(ticks=range(num_rx), labels=range(num_rx))
plt.colorbar()
plt.title("True H Imag (avg over pilot)")
plt.xlabel("TX antenna")
plt.ylabel("RX antenna")

# Imag part heatmap - Predicted
plt.subplot(2, 2, 4)
imag_pred = pred_h[:, :, :, 1].mean(axis=2)
plt.imshow(imag_pred, cmap='viridis', aspect='auto')
plt.xticks(ticks=range(num_tx), labels=range(num_tx))
plt.yticks(ticks=range(num_rx), labels=range(num_rx))
plt.colorbar()
plt.title("Predicted H Imag (avg over pilot)")
plt.xlabel("TX antenna")
plt.ylabel("RX antenna")

plt.suptitle("Channel Matrix (Averaged Over Pilots)")
plt.tight_layout()
plt.savefig(f"{save_dir}/{prefix}_Heatmap.png")

