import os
import scipy.io
import numpy as np
import pickle

# ===== 參數設定 =====
mat_dir = "C:/Users/user/Desktop/2025_resume/CSI_Estimator_Wtih_MMSE/data"  # 根據實際路徑調整
bs_id = 1
user_id = 0  # DeepMIMO user index 從 1 開始
num_tx = 4
num_rx = 4
pilot_length = 8  # 與訓練模型一致
N = 10000  # 資料量，可以根據 path 數量動態決定

# ===== 讀取 CIR 檔案 =====
cir_path = os.path.join(mat_dir, f"O1_60.{bs_id}.CIR.mat")
cir_mat = scipy.io.loadmat(cir_path)
CIR = cir_mat["CIR_array_full"]  # shape: (1, N) or (N, 9)

# ===== 整理 CIR 資料格式 =====
if CIR.ndim == 2 and CIR.shape[0] == 1:
    CIR = CIR[0].reshape(-1, 9)
elif CIR.ndim == 2 and CIR.shape[1] == 9:
    pass
else:
    raise ValueError(f"Unexpected CIR shape: {CIR.shape}")

# ===== 選取特定 user 的 path =====
user_cir = CIR[CIR[:, 0] == (user_id + 1)]  # DeepMIMO user index 從 1 開始
if user_cir.shape[0] == 0:
    raise ValueError(f"No CIR path found for user {user_id}")

# ===== 建立 H 資料集格式： [N, num_rx, num_tx, pilot_length, 2] =====
H = np.zeros((N, num_rx, num_tx, pilot_length, 2), dtype=np.float32)
for i in range(N):
    # 產生隨機 Rayleigh channel（與訓練資料一致）
    h = (np.random.randn(num_rx, num_tx, pilot_length) +
         1j * np.random.randn(num_rx, num_tx, pilot_length)) / np.sqrt(2)
    H[i, ..., 0] = np.real(h)
    H[i, ..., 1] = np.imag(h)

# ===== 儲存為 .pkl 檔案 =====
with open("deepmimo_data.pkl", "wb") as f:
    pickle.dump(H, f)

print("✅ 已產生 deepmimo_data.pkl，shape =", H.shape)
