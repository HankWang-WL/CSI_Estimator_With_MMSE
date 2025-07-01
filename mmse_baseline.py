import torch

def mmse_estimation(x, y, scale=None, noise_var=0.01, eps=1e-4):
    """
    正確版本的 MMSE baseline，支援 Rayleigh 與 DeepMIMO：
    - x, y shape: [B, num_rx, num_tx, pilot_len, 2]
    - scale: 還原用的縮放因子
    """
    x_c = x[..., 0] + 1j * x[..., 1]
    y_c = y[..., 0] + 1j * y[..., 1]
    
    denom = torch.abs(x_c) ** 2 + noise_var
    denom = torch.clamp(denom, min=eps)  # 避免除以 0

    h_hat_c = torch.conj(x_c) * y_c / denom
    h_hat = torch.stack([torch.real(h_hat_c), torch.imag(h_hat_c)], dim=-1)

    if scale is not None:
        view_shape = [x.shape[0]] + [1] * (h_hat.ndim - 1)
        h_hat = h_hat / scale.view(*view_shape)  # 還原回原始尺度

    return h_hat
