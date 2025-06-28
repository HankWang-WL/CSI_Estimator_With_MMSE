import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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

# 畫 h 值分布直方圖
train_h = train_dataset.dataset.H[train_dataset.indices]
val_h = val_dataset.dataset.H[val_dataset.indices]

plt.figure(figsize=(8, 4))
plt.hist(train_h.numpy().flatten(), bins=100, alpha=0.5, label='train')
plt.hist(val_h.numpy().flatten(), bins=100, alpha=0.5, label='val')
plt.legend()
plt.title("h value distribution (train vs val)")
plt.xlabel("h value")
plt.ylabel("count")
#plt.show()

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

# ========== 5. MMSE Baseline ==========
mmse_losses = []
for x, y, h, scale in val_loader:
    h_mmse = mmse_estimation(x, y)
    h_mmse = h_mmse / scale.view(-1, 1, 1, 1, 1)
    loss = criterion(h_mmse, h)
    mmse_losses.append(loss.item())

mse = sum(mmse_losses) / len(mmse_losses)
print(f"MMSE Baseline MSE：{mse:.4f}")

# ========== 6. 畫圖 ==========
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Train Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, marker='s', label='Validation Loss')
plt.hlines(mse, 1, len(train_losses), colors='r', linestyles='dashed', label='MMSE Baseline')
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Performance: Train vs Validation vs MMSE Baseline")
plt.legend()
plt.grid(True)
#plt.show()

# ========== 7. h 和真實 h 的對比圖（只看一筆資料）========== 
x, y, h, scale = next(iter(val_loader))
input_xy = torch.cat([x, y], dim=-1)
input_cnn = input_xy.permute(0, 4, 1, 2, 3).contiguous()

with torch.no_grad():
    pred_h = model(input_cnn)

true_h = h[0].numpy().reshape(num_rx, num_tx, pilot_length, 2)
pred_h = pred_h[0].cpu().numpy().reshape(num_rx, num_tx, pilot_length, 2)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(true_h[0, 0, :, 0],'o-', label='True h real')
plt.plot(pred_h[0, 0, :, 0], 'x--',label='Pred h real')
plt.title('Real part (RX0 - TX0)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(true_h[0, 0, :, 1],'o-', label='True h imag')
plt.plot(pred_h[0, 0, :, 1], 'x--', label='Pred h imag')
plt.title('Imag part (RX0 - TX0)')
plt.legend()
plt.grid(True)

plt.suptitle('Channel Estimation: True vs Predicted (1 sample, RX0-TX0)')
plt.tight_layout()
#plt.show()

#==========  8. 畫 heatmap 圖========== 
plt.figure(figsize=(12, 5))

# Real part heatmap
plt.subplot(2, 2, 1)
plt.imshow(true_h[:, :, :, 0].mean(axis=2), cmap='viridis', aspect='auto')
plt.colorbar()
plt.title("True H Real (avg over pilot)")
plt.xlabel("TX antenna")
plt.ylabel("RX antenna")

plt.subplot(2, 2, 2)
plt.imshow(pred_h[:, :, :, 0].mean(axis=2), cmap='viridis', aspect='auto')
plt.colorbar()
plt.title("Predicted H Real (avg over pilot)")
plt.xlabel("TX antenna")
plt.ylabel("RX antenna")

# Imag part heatmap
plt.subplot(2, 2, 3)
plt.imshow(true_h[:, :, :, 1].mean(axis=2), cmap='viridis', aspect='auto')
plt.colorbar()
plt.title("True H Imag (avg over pilot)")
plt.xlabel("TX antenna")
plt.ylabel("RX antenna")

plt.subplot(2, 2, 4)
plt.imshow(pred_h[:, :, :, 1].mean(axis=2), cmap='viridis', aspect='auto')
plt.colorbar()
plt.title("Predicted H Imag (avg over pilot)")
plt.xlabel("TX antenna")
plt.ylabel("RX antenna")

plt.suptitle("Channel Matrix (Averaged Over Pilots)")
plt.tight_layout()
plt.show()
