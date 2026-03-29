"""Setup: build data pipeline and train the RUL model.

Before running, place the C-MAPSS .txt files in data/raw/:
  train_FD001.txt, test_FD001.txt, RUL_FD001.txt
Download from: https://data.nasa.gov/docs/legacy/CMAPSSData.zip
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    print("=" * 60)
    print("Jet Engine Predictive Maintenance — Setup")
    print("=" * 60)

    print("\n[1/2] Running data pipeline...")
    from tools.data_pipeline import prepare_dataset
    prepare_dataset("FD001")

    print("\n[2/2] Training XGBoost RUL model...")
    from tools.train_model import train
    train(force_prepare=False)

    print("\n" + "=" * 60)
    print("Setup complete! Run: streamlit run ui/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
