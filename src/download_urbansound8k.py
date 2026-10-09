from pathlib import Path
import urllib.request
import soundata

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

DATA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 50)
print("ELEC5305 - UrbanSound8K Downloader")
print("=" * 50)
print(f"Download directory: {DATA_DIR}")

dataset = soundata.initialize(
    "urbansound8k",
    data_home=str(DATA_DIR)
)

print("\nStarting UrbanSound8K download...")

dataset.download()

print("\nDownload operation finished!")
