"""Data loading and preprocessing pipeline for NASA C-MAPSS FD001."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib

from configs.config import (
    DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR,
    COLUMN_NAMES, FEATURE_COLUMNS, RUL_CLIP,
    LOW_VARIANCE_SENSORS
)


def load_cmapss(dataset: str = "FD001") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw train/test splits for a C-MAPSS dataset."""
    raw_dir = DATA_RAW_DIR

    train_path = raw_dir / f"train_{dataset}.txt"
    test_path = raw_dir / f"test_{dataset}.txt"
    rul_path = raw_dir / f"RUL_{dataset}.txt"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {train_path}. "
            "Run: python tools/download_data.py"
        )

    train = pd.read_csv(train_path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    test = pd.read_csv(test_path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    rul_test = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL"])

    return train, test, rul_test


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Add piecewise-linear RUL target column to training data."""
    max_cycles = df.groupby("unit_id")["cycle"].max()
    df = df.merge(max_cycles.rename("max_cycle"), on="unit_id")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df["RUL"] = df["RUL"].clip(upper=RUL_CLIP)
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def add_rolling_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add rolling mean and std features per engine."""
    df = df.copy()
    for col in FEATURE_COLUMNS:
        if col.startswith("sensor_"):
            df[f"{col}_roll_mean"] = (
                df.groupby("unit_id")[col]
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
            df[f"{col}_roll_std"] = (
                df.groupby("unit_id")[col]
                .transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
            )
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = add_rolling_features(df)
    return df


def fit_scaler(train: pd.DataFrame, feature_cols: list[str]) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(train[feature_cols])
    return scaler


def scale(df: pd.DataFrame, scaler: MinMaxScaler, feature_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols])
    return df


def get_feature_columns_with_rolling() -> list[str]:
    """Return full feature list including rolling stats."""
    base = FEATURE_COLUMNS
    rolling = []
    for col in base:
        if col.startswith("sensor_"):
            rolling += [f"{col}_roll_mean", f"{col}_roll_std"]
    return base + rolling


def prepare_dataset(dataset: str = "FD001") -> dict:
    """Full end-to-end data preparation. Returns dict of ready-to-use arrays."""
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_raw, test_raw, rul_test = load_cmapss(dataset)

    # Add RUL to training data
    train = add_rul(train_raw.copy())

    # Feature engineering
    train = build_features(train)
    test = build_features(test_raw.copy())

    feature_cols = get_feature_columns_with_rolling()

    # Fit scaler on train, apply to both
    scaler = fit_scaler(train, feature_cols)
    train = scale(train, scaler, feature_cols)
    test = scale(test, scaler, feature_cols)

    # For test set, take last cycle per engine + add true RUL
    test_last = test.groupby("unit_id").last().reset_index()
    test_last["RUL"] = rul_test["RUL"].values

    # Save artifacts
    train.to_parquet(DATA_PROCESSED_DIR / f"train_{dataset}.parquet", index=False)
    test_last.to_parquet(DATA_PROCESSED_DIR / f"test_{dataset}_last.parquet", index=False)
    test.to_parquet(DATA_PROCESSED_DIR / f"test_{dataset}_full.parquet", index=False)
    joblib.dump(scaler, MODELS_DIR / f"scaler_{dataset}.joblib")

    print(f"[data_pipeline] Train shape: {train.shape}")
    print(f"[data_pipeline] Test (last cycle) shape: {test_last.shape}")
    print(f"[data_pipeline] Feature cols: {len(feature_cols)}")

    return {
        "train": train,
        "test_last": test_last,
        "test_full": test,
        "scaler": scaler,
        "feature_cols": feature_cols,
    }


def load_processed(dataset: str = "FD001") -> dict:
    """Load pre-processed dataset from disk."""
    train = pd.read_parquet(DATA_PROCESSED_DIR / f"train_{dataset}.parquet")
    test_last = pd.read_parquet(DATA_PROCESSED_DIR / f"test_{dataset}_last.parquet")
    test_full = pd.read_parquet(DATA_PROCESSED_DIR / f"test_{dataset}_full.parquet")
    scaler = joblib.load(MODELS_DIR / f"scaler_{dataset}.joblib")
    feature_cols = get_feature_columns_with_rolling()
    return {
        "train": train,
        "test_last": test_last,
        "test_full": test_full,
        "scaler": scaler,
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    prepare_dataset("FD001")
