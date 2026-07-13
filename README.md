# sensor-anomaly-detection
Autoencoder-based anomaly detection on NASA CMAPSS jet engine sensor data | pyTorch | LSTM | Industrial AI

This project focuses on detecting abnormal engine behaviour using sensor time-series data.

The current pipeline includes:

- Loading raw CMAPSS training data
- Removing empty columns
- Computing Remaining Useful Life (RUL)
- Labelling the last 30% of each engine life as anomalous
- Normalising sensor values using min-max scaling

## Tech Stack

- Python
- pandas
- PyTorch
- scikit-learn
- matplotlib
