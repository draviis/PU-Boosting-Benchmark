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


def construct_data(scheme, seed, c=0.2):
    """Use 60/20/20 partitions, then 2/3 background and 1/3 source reservoirs.

    One-sample labels background positives independently with probability c.
    Two-sample matches its P/U counts, drawing P from the separate reservoir
    and U uniformly from the background. Test data are identical across schemes.
    """
    if scheme not in {"one_sample", "two_sample"} or not 0 < c < 1:
        raise ValueError("Invalid sampling scheme or labeling probability")
    x, y, source_ids = load_dataset()
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
        "dataset": os.environ["PU_DATASET"], "positive_sets_nested": True, "sampling_scheme": scheme, "seed": seed, "labeling_probability_one_sample": c,
        "population_prior_supplied": pi, "prior_regime": "oracle_empirical_training_background",
        "test_labels_used_for_prior": False, "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
