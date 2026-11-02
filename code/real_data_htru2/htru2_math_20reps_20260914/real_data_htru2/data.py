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


def construct_data(scheme, seed, c=0.2):
    """Use 60/20/20 partitions, then 2/3 background and 1/3 source reservoirs.

    One-sample labels background positives independently with probability c.
    Two-sample matches its P/U counts, drawing P from the separate reservoir
    and U uniformly from the background. Test data are identical across schemes.
    """
    if scheme not in {"one_sample", "two_sample"} or not 0 < c < 1:
        raise ValueError("Invalid sampling scheme or labeling probability")
    x, y, source_ids = load_htru2()
    indices = np.arange(len(y))
    dev, test = train_test_split(indices, test_size=0.2, stratify=y, random_state=seed)
    train, val = train_test_split(dev, test_size=0.25, stratify=y[dev], random_state=seed + 1)
    rng = np.random.default_rng(seed + 10)
    prepared = {}
    assignments = []
    background_train = None
    for name, partition in (("train", train), ("val", val)):
        background, reservoir = train_test_split(
            partition, test_size=1/3, stratify=y[partition], random_state=seed + (2 if name == "train" else 3)
        )
        if name == "train":
            background_train = background
        rng = np.random.default_rng(seed + (10 if name == "train" else 20))
        s = np.zeros(len(background), dtype=int)
        positive_positions = np.flatnonzero(y[background] == 1)
        s[positive_positions] = (rng.random(len(positive_positions)) < c).astype(int)
        n_p = int(s.sum())
        if n_p < 2:
            raise ValueError("Too few labeled positives; increase the labeling budget")
        # Always make both draws so both schemes have the same random stream.
        positive_source = reservoir[y[reservoir] == 1]
        if n_p > len(positive_source):
            raise ValueError("The separate source contains too few positives")
        p_ids = rng.permutation(positive_source)[:n_p]
        u_ids = rng.permutation(background)[:len(background) - n_p]
        if scheme == "one_sample":
            selected = background
        else:
            selected = np.concatenate([p_ids, u_ids])
            s = np.concatenate([np.ones(n_p, dtype=int), np.zeros(len(u_ids), dtype=int)])
        prepared[name] = {"indices": selected, "s": s}
        selected_labels = dict(zip(selected.tolist(), s.tolist()))
        for reservoir_name, ids in (("background", background), ("positive_source", reservoir)):
            assignments.extend({
                "source_row": int(source_ids[i]), "partition": name,
                "reservoir": reservoir_name, "selected": i in selected_labels,
                "observed_s": selected_labels.get(i, -1), "y_audit_only": int(y[i]),
            } for i in ids)
    prepared["test"] = {"indices": test, "s": np.zeros(len(test), dtype=int)}
    assignments.extend({"source_row": int(source_ids[i]), "partition": "test",
                        "reservoir": "test", "selected": True, "observed_s": -1,
                        "y_audit_only": int(y[i])} for i in test)
    # This is privileged construction information, NOT a PU-estimated prior.
    pi = float(y[background_train].mean())
    scaler = StandardScaler().fit(x[prepared["train"]["indices"]])
    for split in prepared.values():
        ids = split.pop("indices")
        split.update(X=scaler.transform(x[ids]), y=y[ids], source_row=source_ids[ids])
    return prepared, pd.DataFrame(assignments), {
        "dataset": "HTRU2", "positive_sets_nested": True, "sampling_scheme": scheme, "seed": seed, "labeling_probability_one_sample": c,
        "population_prior_supplied": pi, "prior_regime": "oracle_empirical_training_background",
        "test_labels_used_for_prior": False, "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
