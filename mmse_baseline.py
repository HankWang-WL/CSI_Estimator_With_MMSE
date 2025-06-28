import torch

def mmse_estimation(x, y, noise_var=0.01):
    """
    MMSE channel estimation baseline
    適用於 Rayleigh fading + AWGN 通道模型，公式：
        h_hat = conj(x) * y / (|x|^2 + noise_var)
    
    支援兩種輸入格式：
    - x, y shape: [batch, pilot_length, 2]（SISO）
    - x, y shape: [batch, num_rx, num_tx, pilot_length, 2]（MIMO）
    
    返回：
    - h_hat: 預測的通道，shape 與 x, y 相同，最後一維為 [real, imag]
    """
    # 將實數 + 虛數還原為複數形式
    x_c = x[..., 0] + 1j * x[..., 1]
    y_c = y[..., 0] + 1j * y[..., 1]

    # MMSE estimation 公式
    h_hat_c = torch.conj(x_c) * y_c / (torch.abs(x_c)**2 + noise_var)

    # 拆解成 real / imag 結構
    h_hat_real = torch.real(h_hat_c)
    h_hat_imag = torch.imag(h_hat_c)

    # 最後堆疊為 [real, imag] 形式
    h_hat = torch.stack([h_hat_real, h_hat_imag], dim=-1)

    return h_hat
