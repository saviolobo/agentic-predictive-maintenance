"""One-shot setup: download data → build pipeline → train model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    print("=" * 60)
    print("Jet Engine Predictive Maintenance — Setup")
    print("=" * 60)

    # Step 1: Download data
    print("\n[1/3] Downloading NASA C-MAPSS dataset...")
    from tools.download_data import download
    download()

    # Step 2: Build data pipeline
    print("\n[2/3] Running data pipeline...")
    from tools.data_pipeline import prepare_dataset
    prepare_dataset("FD001")

    # Step 3: Train model
    print("\n[3/3] Training XGBoost RUL model...")
    from tools.train_model import train
    train(force_prepare=False)

    print("\n" + "=" * 60)
    print("Setup complete!")
    print("Run the Streamlit app with:")
    print("  streamlit run ui/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
