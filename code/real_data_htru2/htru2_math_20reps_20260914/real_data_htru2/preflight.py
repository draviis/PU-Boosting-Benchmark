"""Check all splits, nested budgets, and train-only scaling before submission."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from data import construct_data, load_htru2

ROOT = Path(__file__).resolve().parent
SEEDS = range(916000, 918000, 100)

def main():
    x, y, ids = load_htru2()
    counts = []
    for seed in SEEDS:
        conditions = {}
        for c in (0.2, 0.05):
            for scheme in ("one_sample", "two_sample"):
                data, audit, meta = construct_data(scheme, seed, c)
                conditions[c, scheme] = data, audit
                split_sets = []
                for name, split in data.items():
                    row_ids = split["source_row"]
                    assert len(set(row_ids)) == len(row_ids)
                    split_sets.append(set(row_ids))
                    assert split["X"].shape[1] == 8 and np.isfinite(split["X"]).all()
                    assert np.all(split["y"][split["s"] == 1] == 1)
                    if name != "test":
                        assert split["s"].sum() >= 2
                    counts.append(dict(seed=seed,c=c,scheme=scheme,split=name,n=len(row_ids),n_P=int(split["s"].sum())))
                assert not any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i))
                np.testing.assert_allclose(data["train"]["X"].mean(axis=0), 0, atol=1e-12)
                for name, split in data.items():
                    expected = (x[np.searchsorted(ids, split["source_row"])] - meta["scaler_mean"]) / meta["scaler_scale"]
                    np.testing.assert_allclose(split["X"], expected)
            one, two = conditions[c,"one_sample"][0], conditions[c,"two_sample"][0]
            np.testing.assert_array_equal(one["test"]["source_row"],two["test"]["source_row"])
            for name in ("train", "val"):
                assert len(one[name]["s"]) == len(two[name]["s"])
                assert one[name]["s"].sum() == two[name]["s"].sum()
        for scheme in ("one_sample", "two_sample"):
            high, low = conditions[0.2,scheme], conditions[0.05,scheme]
            cols = ["source_row","partition","reservoir","y_audit_only"]
            pd.testing.assert_frame_equal(high[1][cols], low[1][cols])
            np.testing.assert_array_equal(high[0]["test"]["source_row"],low[0]["test"]["source_row"])
            for name in ("train", "val"):
                a, b = high[0][name], low[0][name]
                assert set(b["source_row"][b["s"] == 1]) <= set(a["source_row"][a["s"] == 1])
                assert set(a["source_row"][a["s"] == 0]) <= set(b["source_row"][b["s"] == 0])
    table = pd.DataFrame(counts)
    table.to_csv(ROOT / "label_counts.csv", index=False)
    print(table.groupby(["c","scheme","split"]).n_P.agg(["min","median","max"]).to_string())
    (ROOT / "preflight_complete.json").write_text(json.dumps(dict(conditions_checked=80,nested_labels=True,paired_test_sets=True,train_only_scaling=True),indent=2))

if __name__ == "__main__":
    main()
