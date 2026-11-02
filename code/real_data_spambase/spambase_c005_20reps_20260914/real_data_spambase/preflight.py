"""Check all 20 nested split pairs and report actual positive-label budgets."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from low_label_data import construct_data

ROOT = Path(__file__).resolve().parent


def main():
    counts = []
    for seed in range(914000, 916000, 100):
        pairs = {}
        for scheme in ("one_sample", "two_sample"):
            data, audit, meta = construct_data(scheme, seed)
            old = pd.read_csv(ROOT / "reference_c020" / f"{scheme}_seed{seed}" / "assignments_audit_only.csv")
            pd.testing.assert_frame_equal(audit[["source_row","partition","reservoir","y_audit_only"]],old[["source_row","partition","reservoir","y_audit_only"]])
            assert set(audit[audit.observed_s==1].source_row) <= set(old[old.observed_s==1].source_row)
            if scheme == "two_sample":
                assert set(old[old.observed_s==0].source_row) <= set(audit[audit.observed_s==0].source_row)
            for name, split in data.items():
                assert len(set(split["source_row"])) == len(split["source_row"])
                assert np.isfinite(split["X"]).all()
                assert np.all(split["y"][split["s"]==1] == 1)
                counts.append(dict(seed=seed,scheme=scheme,split=name,n=len(split["s"]),n_P=int(split["s"].sum()),n_U=int((split["s"]==0).sum())))
            pairs[scheme] = data
        np.testing.assert_array_equal(pairs["one_sample"]["test"]["source_row"],pairs["two_sample"]["test"]["source_row"])
        for name in ("train","val"):
            assert len(pairs["one_sample"][name]["s"]) == len(pairs["two_sample"][name]["s"])
            assert pairs["one_sample"][name]["s"].sum() == pairs["two_sample"][name]["s"].sum()
    table = pd.DataFrame(counts)
    table.to_csv(ROOT / "low_label_counts.csv",index=False)
    print(table.groupby(["scheme","split"]).n_P.agg(["min","median","max"]).to_string())
    (ROOT / "low_label_preflight.json").write_text(json.dumps(dict(verified_pairs=20,positive_sets_nested=True,partitions_preserved=True),indent=2))


if __name__ == "__main__":
    main()
