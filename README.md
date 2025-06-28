# 📡 CSI Channel Estimation with Deep Learning

This project uses 3D CNN, LSTM, and Transformer architectures to estimate MIMO channel state information (CSI) from pilot signals, and compares them with the MMSE baseline.

---

## 🧠 Key Features

- Support for synthetic and DeepMIMO datasets
- Multiple model types: 3D CNN, LSTM, Transformer
- MMSE baseline comparison
- Full training/validation pipeline with loss plots
- Visualization: heatmap & per-antenna comparison

---

## 📁 Project Structure

CSI_Estimator_Wtih_MMSE/
├── model.py # CNN, LSTM, Transformer modules
├── dataset.py # CSI dataset generator (synthetic + DeepMIMO)
├── mmse_baseline.py # MMSE channel estimator
├── config.py # All config in one place
├── main.py # Training pipeline & evaluation
├── deepmimo_data.pkl # (optional) DeepMIMO-formatted CSI data
└── README.md # This file
