import os
import subprocess
import pprint

combinations = [
    ('cnn', False),
    ('cnn', True),
    ('lstm', False),
    ('lstm', True),
    ('transformer', False),
    ('transformer', True),
]

for model, use_deepmimo in combinations:
    config = {
        "use_deepmimo": use_deepmimo,
        "data_path": "deepmimo_data.pkl",
        "num_rx": 4,
        "num_tx": 4,
        "pilot_length": 8,
        "num_samples": 10000,
        "model_type": model,
        "batch_size": 32,
        "train_epoch": 15,
        "learning_rate": 0.0015,
        "snr_db": "random",
        "quant_bits": 4,
        "exp_decay": 0.5,
        "device": "cuda"
    }

    # 寫入格式良好的 config.py
    with open("config.py", "w") as f:
        f.write("# Auto-generated config. Do not edit manually.\n")
        f.write("config = ")
        pprint.pprint(config, stream=f)

    print(f"\n========== Running: model={model}, use_deepmimo={use_deepmimo} ==========")
    subprocess.run([
        "c:/Users/user/Desktop/2025_resume/CSI_Estimator_Wtih_MMSE/myenv310/Scripts/python.exe", 
        "main.py"
    ])
