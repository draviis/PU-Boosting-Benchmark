"""Independently recompute saved test metrics and check paired data budgets."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             balanced_accuracy_score, matthews_corrcoef,
                             brier_score_loss, log_loss)


def verify(output):
    completed = sorted(output.glob("*/complete.json"))
    if not completed:
        raise ValueError("No completed workers")
    checked = 0
    for marker in completed:
        folder = marker.parent
        results = pd.read_csv(folder / "results.csv", float_precision="round_trip")
        assert len(results) == 27 and results.method_id.is_unique
        assert (results.status == "ok").all()
        assert results.groupby("method_block").size().to_dict() == {"Block0": 2, "Block1": 3, "Block2": 20, "Block3": 2}
        with np.load(folder / "data_audit_only.npz") as saved:
            expected_rows = saved["test_source_row"]
            expected_labels = saved["test_y"]
        for _, row in results.iterrows():
            prediction = pd.read_csv(folder / "predictions" / f"{row.method_id}.csv", float_precision="round_trip")
            np.testing.assert_array_equal(prediction.source_row, expected_rows)
            np.testing.assert_array_equal(prediction.y_true, expected_labels)
            y, score, hard = prediction.y_true, prediction.score, prediction.prediction
            np.testing.assert_array_equal(hard, (score >= row.threshold).astype(int))
            actual = {"AUROC": roc_auc_score(y, score), "AUPRC": average_precision_score(y, score),
                      "F1": f1_score(y, hard, zero_division=0), "BalancedAccuracy": balanced_accuracy_score(y, hard),
                      "MCC": matthews_corrcoef(y, hard)}
            if prediction.probability.notna().all():
                actual.update(Brier=brier_score_loss(y, prediction.probability), LogLoss=log_loss(y, prediction.probability))
            else:
                assert np.isnan(row.Brier) and np.isnan(row.LogLoss)
            for metric, value in actual.items():
                np.testing.assert_allclose(row[metric], value, rtol=1e-8, atol=1e-10)
            checked += 1
        manifest = json.loads((folder / "manifest.json").read_text())
        paired = output / f"two_sample_seed{manifest['seed']}"
        if manifest["sampling_scheme"] == "one_sample" and (paired / "complete.json").exists():
            a = pd.read_csv(folder / "data_counts.csv")
            b = pd.read_csv(paired / "data_counts.csv")
            pd.testing.assert_frame_equal(a[["split", "n", "n_P", "n_U"]], b[["split", "n", "n_P", "n_U"]])
            with np.load(folder / "data_audit_only.npz") as left, np.load(paired / "data_audit_only.npz") as right:
                np.testing.assert_array_equal(left["test_source_row"], right["test_source_row"])
                np.testing.assert_array_equal(left["test_y"], right["test_y"])
    print(f"Verified {checked} method results across {len(completed)} completed workers.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    verify(parser.parse_args().output)
