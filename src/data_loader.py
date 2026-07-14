from pathlib import Path

import pandas as pd
import numpy as np


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


def normalize_sensors(df):
    """
    Normalize sensor columns using min-max scaling.

    Formula:
    x_scaled = (x - min) / (max - min)

    For now, we use training data only.
    Later, the same min/max values will be reused for test data.
    """

    df = df.copy()

    sensor_min = df[SENSOR_COLUMNS].min()
    sensor_max = df[SENSOR_COLUMNS].max()

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
    train_df = normalize_sensors(train_df)

    return train_df



if __name__ == "__main__":
    train_df = load_train_data()

    print("Training data loaded successfully")
    print("Shape:", train_df.shape)
    print()
    print(train_df.head())
    print()
    print("Label distribution:")
    print(train_df["label"].value_counts())
    print()
    print("Sensor value range after normalization:")
    print(train_df[SENSOR_COLUMNS].min().min(), train_df[SENSOR_COLUMNS].max().max())

    X_train_windows, y_train_windows = create_sliding_windows(
        train_df,
        window_size=30,
        step=1
    )

    print()
    print("Sliding windows created successfully")
    print("X_train_windows shape:", X_train_windows.shape)
    print("y_train_windows shape:", y_train_windows.shape)
    print("Window label distribution:")
    print(pd.Series(y_train_windows).value_counts())