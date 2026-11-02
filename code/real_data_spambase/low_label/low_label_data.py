"""Nested c=0.05 samples using the archived c=0.20 assignments.

The old labeled positives are independently retained with probability 0.25.
Two-sample P is reduced to match that count; U is expanded uniformly within
the original background reservoir. Partitions/reservoirs are never regenerated.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from source_data import load_spambase

ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def source_arrays():
    return load_spambase()


def construct_data(scheme, seed, c=0.05):
    if scheme not in {"one_sample", "two_sample"} or c != 0.05:
        raise ValueError("This frozen package is for c=0.05 only")
    references = ROOT / "reference_c020"
    paths = {s: references / f"{s}_seed{seed}" for s in ("one_sample", "two_sample")}
    audits = {s: pd.read_csv(p / "assignments_audit_only.csv") for s, p in paths.items()}
    manifests = {s: json.loads((p / "manifest.json").read_text()) for s, p in paths.items()}
    for manifest in manifests.values():
        assert manifest["seed"] == seed and manifest["labeling_probability_one_sample"] == 0.2
    x, y, source_rows = source_arrays()
    index = {int(row): i for i, row in enumerate(source_rows)}
    selected = {}
    for name, offset in (("train", 3000), ("val", 4000)):
        rng = np.random.default_rng(seed + offset)
        one = audits["one_sample"]
        two = audits["two_sample"]
        background = one.loc[(one.partition == name) & (one.reservoir == "background"), "source_row"].to_numpy()
        old_p_one = one.loc[(one.partition == name) & (one.observed_s == 1), "source_row"].to_numpy()
        kept_p_one = old_p_one[rng.random(len(old_p_one)) < c / 0.2]
        n_p = len(kept_p_one)
        if n_p < 2:
            raise ValueError(f"Insufficient positives: seed={seed}, split={name}, nP={n_p}; do not redraw selectively")
        old_p_two = two.loc[(two.partition == name) & (two.observed_s == 1), "source_row"].to_numpy()
        old_u_two = two.loc[(two.partition == name) & (two.observed_s == 0), "source_row"].to_numpy()
        kept_p_two = rng.choice(old_p_two, n_p, replace=False)
        remaining_background = np.setdiff1d(background, old_u_two)
        extra_u = rng.choice(remaining_background, len(old_p_two) - n_p, replace=False)
        new_u = np.concatenate([old_u_two, extra_u])
        if scheme == "one_sample":
            rows = background
            s = np.isin(rows, kept_p_one).astype(int)
        else:
            rows = np.concatenate([kept_p_two, new_u])
            s = np.concatenate([np.ones(n_p, dtype=int), np.zeros(len(new_u), dtype=int)])
        selected[name] = dict(source_row=rows, s=s)
    test_rows = audits[scheme].loc[audits[scheme].partition == "test", "source_row"].to_numpy()
    selected["test"] = dict(source_row=test_rows, s=np.zeros(len(test_rows), dtype=int))
    train_indices = [index[int(row)] for row in selected["train"]["source_row"]]
    scaler = StandardScaler().fit(x[train_indices])
    for split in selected.values():
        ids = [index[int(row)] for row in split["source_row"]]
        split.update(X=scaler.transform(x[ids]), y=y[ids])
    audit = audits[scheme].copy()
    audit["selected"] = False
    audit["observed_s"] = -1
    for name, split in selected.items():
        row_labels = dict(zip(split["source_row"], split["s"]))
        mask = audit.source_row.isin(row_labels)
        audit.loc[mask, "selected"] = True
        if name != "test":
            audit.loc[mask, "observed_s"] = audit.loc[mask, "source_row"].map(row_labels).astype(int)
    metadata = dict(sampling_scheme=scheme, seed=seed, labeling_probability_one_sample=c,
                    population_prior_supplied=manifests["one_sample"]["population_prior_supplied"],
                    prior_regime="oracle_empirical_training_background", test_labels_used_for_prior=False,
                    label_construction="independent_0.25_thinning_of_archived_c020_labels",
                    reference_c=0.2, reference_fingerprints={s: m["data_fingerprint"] for s,m in manifests.items()},
                    reference_assignment_sha256={s: hashlib.sha256((p/"assignments_audit_only.csv").read_bytes()).hexdigest() for s,p in paths.items()},
                    scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist())
    return selected, audit, metadata
