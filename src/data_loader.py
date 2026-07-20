from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


COLUMN_NAMES = [
    "engine_id",
    "cycle",
    "op1",
    "op2",
    "op3",
    *[f"sensor_{i}" for i in range(1, 22)],
    "nan_1",
    "nan_2",
]

SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]


class SensorWindowDataset(Dataset):
    """
    PyTorch Dataset for sensor sliding windows.

    For training:
        returns only X because autoencoder does not need labels.

    For testing:
        returns X and y because we need labels for evaluation.
    """

    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)

        if y is None:
            self.y = None
        else:
            self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if self.y is None:
            return self.X[index]

        return self.X[index], self.y[index]


def load_cmapss_file(file_path):
    """
    Load NASA CMAPSS FD001 file.

    The dataset is space-separated and has no header.
    Extra empty columns are dropped after loading.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES
    )

    df = df.dropna(axis=1, how="all")

    return df


def add_rul(df):
    """
    Add Remaining Useful Life.

    RUL = max cycle of engine - current cycle
    """

    df = df.copy()

    max_cycle = df.groupby("engine_id")["cycle"].transform("max")

    df["max_cycle"] = max_cycle
    df["RUL"] = df["max_cycle"] - df["cycle"]

    return df


def add_test_rul(test_df, rul_file_path):
    """
    Add RUL for test data using RUL_FD001.txt.

    Test engines do not run until failure.
    RUL_FD001.txt gives the remaining cycles after the last observed cycle.
    """

    test_df = test_df.copy()

    rul_df = pd.read_csv(
        rul_file_path,
        sep=r"\s+",
        header=None,
        names=["final_RUL"]
    )

    rul_df["engine_id"] = range(1, len(rul_df) + 1)

    max_observed_cycle = test_df.groupby("engine_id")["cycle"].max().reset_index()
    max_observed_cycle = max_observed_cycle.rename(columns={"cycle": "last_observed_cycle"})

    test_df = test_df.merge(max_observed_cycle, on="engine_id", how="left")
    test_df = test_df.merge(rul_df, on="engine_id", how="left")

    test_df["max_cycle"] = test_df["last_observed_cycle"] + test_df["final_RUL"]
    test_df["RUL"] = test_df["max_cycle"] - test_df["cycle"]

    test_df = test_df.drop(columns=["last_observed_cycle", "final_RUL"])

    return test_df


def add_anomaly_label(df, anomaly_ratio=0.30):
    """
    Label last 30 percent of each engine life as anomaly.

    label = 0 means healthy
    label = 1 means anomaly
    """

    df = df.copy()

    anomaly_threshold = anomaly_ratio * df["max_cycle"]

    df["label"] = (df["RUL"] <= anomaly_threshold).astype(int)

    return df


# def normalize_sensors(df):
#     """
#     Normalize sensor columns using min-max scaling.

#     Formula:
#     x_scaled = (x - min) / (max - min)

#     For now, we use training data only.
#     Later, the same min/max values will be reused for test data.
#     """

#     df = df.copy()

#     sensor_min = df[SENSOR_COLUMNS].min()
#     sensor_max = df[SENSOR_COLUMNS].max()

#     denominator = sensor_max - sensor_min
#     denominator = denominator.replace(0, 1)

#     df[SENSOR_COLUMNS] = (df[SENSOR_COLUMNS] - sensor_min) / denominator

#     return df


def fit_sensor_scaler(train_df):
    """
    Compute min and max values from training data only.

    These values will later be used to normalize both training and test data.
    """

    sensor_min = train_df[SENSOR_COLUMNS].min()
    sensor_max = train_df[SENSOR_COLUMNS].max()

    return sensor_min, sensor_max


def apply_sensor_scaler(df, sensor_min, sensor_max):
    """
    Apply min-max normalization using training data statistics.

    This avoids data leakage from the test set.
    """

    df = df.copy()

    denominator = sensor_max - sensor_min
    denominator = denominator.replace(0, 1)

    df[SENSOR_COLUMNS] = (df[SENSOR_COLUMNS] - sensor_min) / denominator

    return df



def create_sliding_windows(df, window_size=30, step=1):
    """
    Create sliding windows from sensor data.

    Each window contains 30 cycles of sensor readings.
    Windows are created separately for each engine, so data from different
    engines is never mixed.

    The label of a window is the label of its last cycle.
    """

    X_windows = []
    y_windows = []

    for engine_id, engine_df in df.groupby("engine_id"):
        engine_df = engine_df.sort_values("cycle").reset_index(drop=True)

        sensor_values = engine_df[SENSOR_COLUMNS].values
        labels = engine_df["label"].values

        for start_idx in range(0, len(engine_df) - window_size + 1, step):
            end_idx = start_idx + window_size

            window = sensor_values[start_idx:end_idx]
            window_label = labels[end_idx - 1]

            X_windows.append(window)
            y_windows.append(window_label)

    X = np.array(X_windows, dtype=np.float32)
    y = np.array(y_windows, dtype=np.int64)

    return X, y


def get_healthy_windows(X, y):
    """
    Keep only healthy windows for autoencoder training.

    label 0 = healthy
    label 1 = anomaly
    """

    healthy_mask = y == 0

    X_healthy = X[healthy_mask]
    y_healthy = y[healthy_mask]

    return X_healthy, y_healthy                     




def load_train_data(raw_data_dir="data/raw"):
    """
    Load training data and prepare basic columns:
    RUL and anomaly label.
    """

    raw_data_dir = Path(raw_data_dir)

    train_path = raw_data_dir / "train_FD001.txt"

    train_df = load_cmapss_file(train_path)
    train_df = add_rul(train_df)
    train_df = add_anomaly_label(train_df)
    # train_df = normalize_sensors(train_df)

    return train_df


def load_test_data(raw_data_dir="data/raw"):
    """
    Load test data and prepare RUL and anomaly labels.
    """

    raw_data_dir = Path(raw_data_dir)

    test_path = raw_data_dir / "test_FD001.txt"
    rul_path = raw_data_dir / "RUL_FD001.txt"

    test_df = load_cmapss_file(test_path)
    test_df = add_test_rul(test_df, rul_path)
    test_df = add_anomaly_label(test_df)

    return test_df


def load_train_test_data(raw_data_dir="data/raw"):
    """
    Load train and test data.

    Normalization is fitted only on training data,
    then applied to both train and test data.
    """

    train_df = load_train_data(raw_data_dir)
    test_df = load_test_data(raw_data_dir)

    sensor_min, sensor_max = fit_sensor_scaler(train_df)

    train_df = apply_sensor_scaler(train_df, sensor_min, sensor_max)
    test_df = apply_sensor_scaler(test_df, sensor_min, sensor_max)

    return train_df, test_df



def prepare_window_data(raw_data_dir="data/raw", window_size=30, step=1):
    """
    Prepare final window-based datasets for anomaly detection.

    X_train:
        Healthy training windows only.

    X_test:
        All test windows.

    y_test:
        Labels for test windows.
    """

    train_df, test_df = load_train_test_data(raw_data_dir)

    X_train_windows, y_train_windows = create_sliding_windows(
        train_df,
        window_size=window_size,
        step=step
    )

    X_train, _ = get_healthy_windows(
        X_train_windows,
        y_train_windows
    )

    X_test, y_test = create_sliding_windows(
        test_df,
        window_size=window_size,
        step=step
    )

    return X_train, X_test, y_test


def create_dataloaders(
    raw_data_dir="data/raw",
    window_size=30,
    step=1,
    batch_size=64
):
    """
    Create PyTorch DataLoaders for model training and evaluation.

    train_loader:
        Contains only healthy training windows.

    test_loader:
        Contains all test windows and their labels.
    """

    X_train, X_test, y_test = prepare_window_data(
        raw_data_dir=raw_data_dir,
        window_size=window_size,
        step=step
    )

    train_dataset = SensorWindowDataset(X_train)
    test_dataset = SensorWindowDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = create_dataloaders(
        raw_data_dir="data/raw",
        window_size=30,
        step=1,
        batch_size=64
    )

    print("PyTorch DataLoaders created successfully")

    train_batch = next(iter(train_loader))
    test_batch_X, test_batch_y = next(iter(test_loader))

    print()
    print("Train batch shape:", train_batch.shape)

    print()
    print("Test batch X shape:", test_batch_X.shape)
    print("Test batch y shape:", test_batch_y.shape)