from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
FIGURE_DIR = ROOT / "reports" / "figures"
RANDOM_STATE = 42
TEST_SIZE = 0.20

for folder in (DATA_RAW, DATA_PROCESSED, MODEL_DIR, FIGURE_DIR):
    folder.mkdir(parents=True, exist_ok=True)
