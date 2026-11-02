"""Read immutable prepared features; split, hide labels and scale per worker."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent

def load_dataset():
    name = os.environ["PU_DATASET"]
    if name not in ("rice","phishing","credit_default"):
        raise ValueError("Unknown dataset")
    with np.load(ROOT / "data" / name / "prepared.npz") as data:
        return data["X"], data["y"], data["source_ids"]
