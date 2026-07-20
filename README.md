# Sensor Anomaly Detection

Autoencoder-based anomaly detection on NASA CMAPSS FD001 jet engine sensor data using Python, PyTorch, and time-series modelling.

This project focuses on detecting abnormal engine behaviour using multivariate sensor time-series data. The goal is to build a complete anomaly detection pipeline for predictive maintenance and compare different modelling approaches.

## Current Pipeline

The current data pipeline includes:

* Loading raw NASA CMAPSS FD001 training and test files
* Removing empty columns from raw text files
* Computing Remaining Useful Life (RUL)
* Using `RUL_FD001.txt` to calculate correct RUL values for test engines
* Labelling the last 30% of each engine life as anomalous
* Normalising sensor values using min-max scaling based only on training data
* Creating sliding windows of 30 cycles for time-series modelling
* Preparing healthy-only training windows for autoencoder training
* Preparing test windows with labels for evaluation
* Wrapping window data into PyTorch Dataset and DataLoader objects

## Dataset

The project uses the NASA CMAPSS FD001 turbofan engine degradation dataset.

Each row represents one engine at one cycle and contains:

* Engine ID
* Cycle number
* 3 operational settings
* 21 sensor measurements

## Modelling Plan

The project will compare three anomaly detection approaches:

1. Linear Autoencoder
2. LSTM Autoencoder
3. Isolation Forest

The autoencoders will be trained only on healthy engine behaviour. During evaluation, high reconstruction error will be used as an anomaly signal.


## Tech Stack

* Python
* pandas
* NumPy
* PyTorch
* scikit-learn
* matplotlib

