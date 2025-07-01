import os
import numpy as np
import scipy.io
import pickle

# =================== 參數設定 ===================
mat_dir = "./data"
bs_id = 1
pilot_length = 8
fc = 60e9  # Hz
delta_f = 15e3  # Hz
num_rx = 4
num_tx = 4
N = 10000
rx_spacing = 0.5
tx_spacing = 0.5
c = 3e8
lambda_c = c / fc
subcarrier_index = np.arange(pilot_length)

# =================== 載入 CIR 資料 ===================
mat_path = os.path.join(mat_dir, f"O1_60.{bs_id}.CIR.mat")
mat = scipy.io.loadmat(mat_path)
CIR = mat['CIR_array_full'].flatten()

n_col = 9
usable_len = len(CIR) - (len(CIR) % n_col)
CIR_data = CIR[:usable_len].reshape(-1, n_col)
user_ids = np.unique(CIR_data[:, 1])[:N]

H = np.zeros((len(user_ids), num_rx, num_tx, pilot_length), dtype=np.complex64)

# =================== 合成通道 ===================
for i, uid in enumerate(user_ids):
    user_cir = CIR_data[CIR_data[:, 1] == uid]
    if len(user_cir) == 0:
        continue
    user_cir = user_cir[np.argsort(-user_cir[:, 6])[:10]]  # 取前 10 條 strongest path

    for rx in range(num_rx):
        for tx in range(num_tx):
            h_sum = np.zeros(pilot_length, dtype=np.complex64)
            for path in user_cir:
                delay = path[4]
                phase = path[5]
                power = path[6]
                AoD_az = path[7] * np.pi / 180
                AoA_az = path[8] * np.pi / 180

                tx_steer = np.exp(1j * 2 * np.pi * tx_spacing * tx * np.sin(AoD_az) / lambda_c)
                rx_steer = np.exp(1j * 2 * np.pi * rx_spacing * rx * np.sin(AoA_az) / lambda_c)

                # 新增隨機相位差，讓每個 subcarrier 有變化
                rand_phase = np.exp(1j * 2 * np.pi * np.random.rand())
                freq_response = np.exp(-1j * 2 * np.pi * delta_f * subcarrier_index * delay)
                a = freq_response * np.exp(1j * phase) * rand_phase * np.sqrt(power)
                a *= tx_steer * rx_steer
                h_sum += a

            h_sum /= np.sqrt(10)
            H[i, rx, tx, :] = h_sum

# =================== 儲存 ===================
H_out = np.stack([np.real(H), np.imag(H)], axis=-1)
with open("deepmimo_data.pkl", "wb") as f:
    pickle.dump([{'user': {'channel': H_out}}], f)

print(f"\n 已產生 {H_out.shape[0]} 筆樣本，通道 shape: {H_out.shape}，已儲存為 deepmimo_data.pkl\n")
