"""Official HTRU2 data, cached with provenance and duplicate auditing."""
import hashlib
import io
import json
import os
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
URL = "https://archive.ics.uci.edu/static/public/372/htru2.zip"

def load_htru2():
    directory = ROOT / "data"
    directory.mkdir(exist_ok=True)
    archive = directory / "htru2.zip"
    read_only = os.environ.get("PU_HTRU2_READ_ONLY_DATA") == "1"
    if not archive.exists():
        if read_only:
            raise FileNotFoundError("Prepare HTRU2 before starting workers")
        with urlopen(URL, timeout=60) as response:
            archive.write_bytes(response.read())
    with ZipFile(archive) as bundle:
        frame = pd.read_csv(io.BytesIO(bundle.read("HTRU_2.csv")), header=None)
    assert frame.shape == (17898, 9)
    assert np.isfinite(frame.to_numpy()).all()
    assert frame[8].value_counts().to_dict() == {0: 16259, 1: 1639}
    features = list(range(8))
    conflicts = frame.groupby(features, dropna=False)[8].transform("nunique") > 1
    cleaned = frame.loc[~conflicts].drop_duplicates(subset=features).copy()
    provenance = dict(source_url=URL, dataset_doi="10.24432/C5DK6R", license="CC BY 4.0",
                      archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                      original_rows=len(frame), conflicting_feature_rows_removed=int(conflicts.sum()),
                      duplicate_rows_removed=int((~conflicts).sum()-len(cleaned)), clean_rows=len(cleaned),
                      features=8, positive_label="pulsar=1", positive_rows=int(cleaned[8].sum()))
    if not read_only:
        (directory / "provenance.json").write_text(json.dumps(provenance, indent=2))
        cleaned.to_csv(directory / "htru2_clean.csv", index_label="source_row")
    return cleaned[features].to_numpy(float), cleaned[8].to_numpy(int), cleaned.index.to_numpy()
