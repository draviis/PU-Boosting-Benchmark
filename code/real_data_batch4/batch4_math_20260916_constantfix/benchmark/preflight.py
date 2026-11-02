"""Audit every seed and condition before launching any model fitting."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from data import construct_data, load_dataset

ROOT=Path(__file__).resolve().parent
DATASETS={"dry_bean":924000,"optical_digits":926000,"online_shoppers":928000}

def main():
    counts=[]
    for dataset,start in DATASETS.items():
        os.environ["PU_DATASET"]=dataset
        x,y,ids=load_dataset()
        for seed in range(start,start+2000,100):
            conditions={}
            for c in (.2,.05):
                for scheme in ("one_sample","two_sample"):
                    data,audit,meta=construct_data(scheme,seed,c)
                    conditions[c,scheme]=data,audit
                    if dataset=="optical_digits":
                        with np.load(ROOT/"data/optical_digits/prepared.npz") as source:
                            expected_test=source["source_ids"][source["is_official_test"]]
                        np.testing.assert_array_equal(data["test"]["source_row"],expected_test)
                    groups=[]
                    for split,values in data.items():
                        rowids=values["source_row"]
                        groups.append(set(rowids))
                        assert len(set(rowids))==len(rowids)
                        assert np.isfinite(values["X"]).all()
                        assert np.all(values["y"][values["s"]==1]==1)
                        if split!="test":
                            assert values["s"].sum()>=2
                        expected=(x[np.searchsorted(ids,rowids)]-meta["scaler_mean"])/meta["scaler_scale"]
                        np.testing.assert_allclose(values["X"],expected)
                        counts.append(dict(dataset=dataset,c=c,scheme=scheme,seed=seed,split=split,n=len(rowids),n_P=int(values["s"].sum())))
                    assert all(not groups[i]&groups[j] for i in range(3) for j in range(i))
                    np.testing.assert_allclose(data["train"]["X"].mean(axis=0),0,atol=1e-10)
                a,b=conditions[c,"one_sample"][0],conditions[c,"two_sample"][0]
                np.testing.assert_array_equal(a["test"]["source_row"],b["test"]["source_row"])
                for split in ("train","val"):
                    assert len(a[split]["s"])==len(b[split]["s"])
                    assert a[split]["s"].sum()==b[split]["s"].sum()
            for scheme in ("one_sample","two_sample"):
                a,b=conditions[.2,scheme],conditions[.05,scheme]
                cols=["source_row","partition","reservoir","y_audit_only"]
                pd.testing.assert_frame_equal(a[1][cols],b[1][cols])
                assert set(b[1].loc[b[1].observed_s==1,"source_row"])<=set(a[1].loc[a[1].observed_s==1,"source_row"])
                assert set(a[1].loc[a[1].observed_s==0,"source_row"])<=set(b[1].loc[b[1].observed_s==0,"source_row"])
                np.testing.assert_array_equal(a[0]["test"]["source_row"],b[0]["test"]["source_row"])
        print("Passed:",dataset,flush=True)
    table=pd.DataFrame(counts)
    table.to_csv(ROOT/"label_counts.csv",index=False)
    print(table.query("split!='test'").groupby(["dataset","c","scheme","split"]).n_P.agg(["min","median","max"]).to_string(),flush=True)
    (ROOT/"preflight_complete.json").write_text(json.dumps(dict(datasets=list(DATASETS),conditions_checked=240,paired_tests=True,nested_labels=True,train_only_scaling=True),indent=2))

if __name__=="__main__":
    main()
