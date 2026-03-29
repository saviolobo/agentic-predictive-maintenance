"""Download and extract NASA C-MAPSS dataset."""
import urllib.request
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
ZIP_PATH = RAW_DIR / "CMAPSSData.zip"


def download():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if ZIP_PATH.exists():
        print(f"[download] Zip already exists at {ZIP_PATH}, skipping download.")
    else:
        print(f"[download] Downloading from {URL} ...")
        urllib.request.urlretrieve(URL, ZIP_PATH)
        print(f"[download] Saved to {ZIP_PATH}")

    print("[download] Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(RAW_DIR)
        print(f"[download] Extracted: {z.namelist()}")

    print("[download] Done.")


if __name__ == "__main__":
    download()
