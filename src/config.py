# src/config.py
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Audio preprocessing
SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 400
HOP_LENGTH = 160
WINDOW = "hann"
FMIN = 80
FMAX = 7600

# Model
EMBEDDING_DIM = 192
DEVICE = "mps"
PRETRAINED_MODEL_NAME = "speechbrain/spkrec-ecapa-voxceleb"

# Training
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 50
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Calibration & threshold
UNCERTAIN_MIN_PROB = 0.25
UNCERTAIN_MAX_PROB = 0.75

# Paths for dataset CSVs
TRAIN_PAIRS_CSV = METADATA_DIR / "train_pairs.csv"
VAL_PAIRS_CSV = METADATA_DIR / "val_pairs.csv"
TEST_PAIRS_CSV = METADATA_DIR / "test_pairs.csv"

print(f"Using device: {DEVICE}")