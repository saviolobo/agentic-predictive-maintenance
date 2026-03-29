"""Train XGBoost RUL prediction model on C-MAPSS FD001."""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from configs.config import MODELS_DIR
from tools.data_pipeline import prepare_dataset, load_processed, get_feature_columns_with_rolling

MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "xgb_rul_FD001.joblib"


def train(dataset: str = "FD001", force_prepare: bool = False) -> XGBRegressor:
    processed_path = Path(f"data/processed/train_{dataset}.parquet")
    if not processed_path.exists() or force_prepare:
        print("[train] Running data preparation pipeline...")
        data = prepare_dataset(dataset)
    else:
        print("[train] Loading pre-processed data...")
        data = load_processed(dataset)

    train_df = data["train"]
    test_df = data["test_last"]
    feature_cols = data["feature_cols"]

    X_train = train_df[feature_cols].values
    y_train = train_df["RUL"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["RUL"].values

    print(f"[train] X_train: {X_train.shape}, y_train: {y_train.shape}")

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20,
        eval_metric="rmse",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\n[train] Test MAE:  {mae:.2f} cycles")
    print(f"[train] Test RMSE: {rmse:.2f} cycles")

    # Save
    joblib.dump(model, MODEL_PATH)
    print(f"[train] Model saved to {MODEL_PATH}")

    return model


def load_model() -> XGBRegressor:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run: python tools/train_model.py"
        )
    return joblib.load(MODEL_PATH)


if __name__ == "__main__":
    train()
