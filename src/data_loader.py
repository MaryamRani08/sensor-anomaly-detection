from pathlib import Path

import pandas as pd


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