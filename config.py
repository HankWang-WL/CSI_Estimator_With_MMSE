config = {
    # ========== 資料來源 ==========
    "use_deepmimo": True,  # True: 使用 DeepMIMO, False: 自創模擬資料
    "data_path": "deepmimo_data.pkl",  # 若使用 DeepMIMO，這裡是 .pkl 路徑

    # ========== 通道參數 ==========
    "num_rx": 4,           # 接收天線數量
    "num_tx": 4,           # 發送天線數量
    "pilot_length": 8,     # pilot 長度（Zadoff-Chu）
    "num_samples": 10000,  # 自創模擬樣本數（DeepMIMO不需設）

    # ========== 模型選擇 ==========
    "model_type": "lstm",   # cnn / lstm / transformer

    # ========== 訓練參數 ==========
    "batch_size": 32,
    "train_epoch": 15,
    "learning_rate": 0.0015,

    # ========== SNR 設定 ==========
    "snr_db": "random",  # 可設固定如 20 或用 "random" 表示每筆隨機 10~30 dB

    # ========== 雜訊與通道失真 ==========
    "quant_bits": 4,     # 複數量化位元數，None 表示不量化
    "exp_decay": 0.5,    # Rayleigh 通道指數衰減係數

    # ========== 裝置與種子 ==========
    "device": "cuda",    # 或 "cpu"
    "seed": 42,
}
