import torch
import numpy as np
import pickle

class CSIDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=10000, pilot_length=8, snr_db="random",
                 exp_decay=0.5, quant_bits=4, num_tx=4, num_rx=4, data_path=None):
        self.snr_db = snr_db
        self.quant_bits = quant_bits
        self.num_tx = num_tx
        self.num_rx = num_rx
        self.pilot_length = pilot_length
        self.exp_decay = exp_decay

        X, Y, H, scales = [], [], [], []

        # 如果有 data_path 就讀取 DeepMIMO
        if data_path is not None:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)

            H_list = []
            for d in data:
                if isinstance(d, dict) and 'user' in d and 'channel' in d['user']:
                    h = np.asarray(d['user']['channel'])
                elif isinstance(d, dict) and 'channel' in d:
                    h = np.asarray(d['channel'])
                else:
                    h = np.asarray(d)
                if h.ndim == 2:
                    h = h[..., np.newaxis]
                H_list.append(h)

            if len(H_list) == 0:
                raise ValueError("你的 DeepMIMO pickle 檔裡沒有任何 channel 資料！")

            for h in H_list:
                if h.shape[-1] == 2:
                    h_complex = h[..., 0] + 1j * h[..., 1]
                    pilot_len = h.shape[-2]
                else:
                    h_complex = h
                    pilot_len = h.shape[-1]

                # Zadoff-Chu pilot
                root = np.random.randint(1, pilot_len)
                pilot = self.generate_zadoff_chu(pilot_len, root)
                pilot_pair = np.stack([np.real(pilot), np.imag(pilot)], axis=1)

                # broadcast pilot
                expand_shape = [1] * (h_complex.ndim - 1) + [pilot_len]
                pilot_broadcast = pilot.reshape(*expand_shape)
                y = h_complex * pilot_broadcast

                # 隨機 SNR
                snr_db_sample = np.random.uniform(10, 30) if self.snr_db == "random" else self.snr_db
                snr_linear = 10 ** (snr_db_sample / 10)
                signal_power = np.mean(np.abs(y)**2)
                noise_power = signal_power / snr_linear
                noise_std = np.sqrt(noise_power / 2)
                noise = np.random.randn(*y.shape) * noise_std + 1j * np.random.randn(*y.shape) * noise_std
                y_noisy = y + noise

                # IQ imbalance + 量化
                y_noisy = self.add_iq_imbalance(y_noisy)
                if self.quant_bits is not None:
                    y_noisy = self.quantize_complex(y_noisy, bits=self.quant_bits)

                h_pair = np.stack([np.real(h_complex), np.imag(h_complex)], axis=-1)
                y_pair = np.stack([np.real(y_noisy), np.imag(y_noisy)], axis=-1)
                x_pair = np.tile(pilot_pair, (*h_pair.shape[:-2], 1, 1))

                h_pair, scale = self.normalize_channel(h_pair, return_scale=True)
                y_pair = self.normalize_channel(y_pair)
                x_pair = self.normalize_channel(x_pair)

                X.append(x_pair)
                Y.append(y_pair)
                H.append(h_pair)
                scales.append(scale)

        else:
            # 自創 Rayleigh channel 模擬
            for _ in range(num_samples):
                x_sample = np.zeros((self.num_rx, self.num_tx, self.pilot_length, 2), dtype=np.float32)
                y_sample = np.zeros_like(x_sample)
                h_sample = np.zeros_like(x_sample)

                for rx in range(self.num_rx):
                    for tx in range(self.num_tx):
                        root = np.random.randint(1, self.pilot_length)
                        pilot = self.generate_zadoff_chu(self.pilot_length, root)
                        h = self.generate_rayleigh_channel(self.pilot_length)
                        y_clean = pilot * h

                        snr_db_sample = np.random.uniform(10, 30) if self.snr_db == "random" else self.snr_db
                        snr_linear = 10 ** (snr_db_sample / 10)
                        signal_power = np.mean(np.abs(y_clean)**2)
                        noise_power = signal_power / snr_linear
                        noise_std = np.sqrt(noise_power / 2)
                        noise = np.random.randn(self.pilot_length) * noise_std + 1j * np.random.randn(self.pilot_length) * noise_std
                        y = y_clean + noise

                        y = self.add_iq_imbalance(y)
                        if self.quant_bits is not None:
                            y = self.quantize_complex(y, bits=self.quant_bits)

                        h_pair = np.stack([np.real(h), np.imag(h)], axis=1)
                        h_pair = self.normalize_channel(h_pair)
                        x_pair = np.stack([np.real(pilot), np.imag(pilot)], axis=1)
                        y_pair = np.stack([np.real(y), np.imag(y)], axis=1)

                        x_sample[rx, tx] = x_pair
                        y_sample[rx, tx] = y_pair
                        h_sample[rx, tx] = h_pair

                X.append(x_sample)
                Y.append(y_sample)
                H.append(h_sample)

            scales = np.ones(len(X), dtype=np.float32)

        # 最終轉換為 tensor
        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        self.Y = torch.tensor(np.array(Y), dtype=torch.float32)
        self.H = torch.tensor(np.array(H), dtype=torch.float32)
        self.scales = torch.tensor(np.array(scales), dtype=torch.float32)

    def __len__(self):
        return self.H.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.H[idx], self.scales[idx]

    def normalize_channel(self, h, return_scale=False):
        h_complex = h[..., 0] + 1j * h[..., 1]
        max_abs = np.max(np.abs(h_complex)) + 1e-12
        h_norm = h / max_abs
        return (h_norm, max_abs) if return_scale else h_norm

    def quantize_complex(self, y, bits=1):
        y_real = np.real(y)
        y_imag = np.imag(y)
        max_val = max(np.max(np.abs(y_real)), np.max(np.abs(y_imag)), 1e-12)
        levels = 2 ** bits
        step = 2 * max_val / (levels - 1)
        yq_real = np.round((y_real + max_val) / step)
        yq_imag = np.round((y_imag + max_val) / step)
        yq = (yq_real - (levels // 2)) * step + 1j * (yq_imag - (levels // 2)) * step
        return yq

    def add_iq_imbalance(self, y, amp_range=0.05, phase_range=5):
        amp_imbalance = 1 + np.random.uniform(-amp_range, amp_range)
        phase_imbalance = np.deg2rad(np.random.uniform(-phase_range, phase_range))
        y_real = np.real(y) * amp_imbalance
        y_imag = np.imag(y)
        y_real_final = y_real
        y_imag_final = y_imag * np.cos(phase_imbalance) - y_real * np.sin(phase_imbalance)
        return y_real_final + 1j * y_imag_final

    def generate_zadoff_chu(self, length, root=1):
        n = np.arange(length)
        chu = np.exp(-1j * np.pi * root * n * (n + 1) / length)
        return chu

    def generate_rayleigh_channel(self, length):
        power_profile = np.exp(-self.exp_decay * np.arange(length))
        power_profile /= np.sum(power_profile)
        real = np.random.randn(length) * np.sqrt(power_profile / 2)
        imag = np.random.randn(length) * np.sqrt(power_profile / 2)
        h = real + 1j * imag
        return h
