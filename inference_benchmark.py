import torch
import time
import numpy as np
from model import SimpleCSINet3D, LSTMCSINet, TransformerCSINet
from dataset import CSIDataset

# ==== 設定參數 ====
USE_DEEPMIMO = False
num_tx = 4
num_rx = 4
pilot_length = 8
BATCH_SIZES = [1, 8, 16, 32]
N_RUNS = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==== 載入資料集（僅用部分資料做推論測試）====
if USE_DEEPMIMO:
    dataset = CSIDataset(
        data_path="deepmimo_data.pkl",
        num_rx=num_rx,
        num_tx=num_tx,
        pilot_length=pilot_length
    )
else:
    dataset = CSIDataset(
        num_samples=1000,
        pilot_length=pilot_length,
        num_tx=num_tx,
        num_rx=num_rx
    )

# ==== 模型列表 ====
model_list = [
    ("CNN", SimpleCSINet3D(num_rx=num_rx, num_tx=num_tx, pilot_length=pilot_length, out_dim2=2)),
    ("LSTM", LSTMCSINet(num_rx=num_rx, num_tx=num_tx, pilot_length=pilot_length, out_dim2=2)),
    ("Transformer", TransformerCSINet(num_rx=num_rx, num_tx=num_tx, pilot_length=pilot_length, out_dim2=2))
]

# ==== 推論時間測試函數 ====
def benchmark_model(model, x, device, n_runs=100):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        # 預熱 CUDA
        for _ in range(10):
            model(x)
            if device == 'cuda':
                torch.cuda.synchronize()
        # 正式量測
        start = time.time()
        for _ in range(n_runs):
            model(x)
            if device == 'cuda':
                torch.cuda.synchronize()
        end = time.time()
        avg_time = (end - start) / n_runs * 1000  # 毫秒
    return avg_time

# ==== 執行 Benchmark ====
results = []
for batch_size in BATCH_SIZES:
    x, y, h, scale = dataset[0]
    x = x.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
    y = y.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)

    input_xy = torch.cat([x, y], dim=-1)  # [B, num_rx, num_tx, pilot_length, 4]
    input_cnn = input_xy.permute(0, 4, 1, 2, 3).contiguous().to(DEVICE)

    for model_name, model in model_list:
        avg_time = benchmark_model(model, input_cnn, DEVICE, n_runs=N_RUNS)
        results.append({
            "Model": model_name,
            "Batch Size": batch_size,
            "Device": DEVICE,
            "Avg Inference Time (ms)": avg_time
        })
        print(f"{model_name} | Batch={batch_size} | Device={DEVICE} | Avg Inference Time: {avg_time:.3f} ms")

# ==== 總結結果 ====
print("\n=== Inference Benchmark Results ===")
print(f"{'Model':<15} {'Batch Size':<12} {'Device':<8} {'Avg Time (ms)':<15}")
for r in results:
    print(f"{r['Model']:<15} {r['Batch Size']:<12} {r['Device']:<8} {r['Avg Inference Time (ms)']:<15.3f}")
