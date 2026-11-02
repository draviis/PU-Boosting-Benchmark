"""Prepare three predeclared binary tasks using the frozen PU benchmark."""
import hashlib
import io
import json
from pathlib import Path
import shutil
from zipfile import ZipFile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent
BUNDLE=ROOT/"batch4_math_20260915"
SOURCE=REPO/"real_data_batch3/batch3_math_20260914"
SPECS={
    "dry_bean":dict(uci_id=602,n=13611,target="Class",seed=924000),
    "optical_digits":dict(uci_id=80,n=5620,target="class",seed=926000),
    "online_shoppers":dict(uci_id=468,n=12330,target="Revenue",seed=928000),
}

def prepare_dataset(name,spec,code):
    folder=code/"data"/name
    folder.mkdir(parents=True)
    source=ROOT/"data"/f"{spec['uci_id']}.csv"
    metadata=ROOT/"data"/f"{spec['uci_id']}_metadata.json"
    shutil.copy2(source,folder/"source.csv")
    shutil.copy2(metadata,folder/"uci_metadata.json")
    frame=pd.read_csv(source)
    assert len(frame)==spec["n"] and not frame.isna().any().any()
    original_target=frame.pop(spec["target"])
    fixed_test=np.zeros(len(frame),dtype=bool)
    if name=="dry_bean":
        y=(original_target=="DERMASON").astype(int)
        definition="Class=DERMASON versus all six other varieties"
    elif name=="optical_digits":
        archive=ROOT/"data/optdigits.zip"
        shutil.copy2(archive,folder/"original_archive.zip")
        with ZipFile(archive) as z:
            train=pd.read_csv(io.BytesIO(z.read("optdigits.tra")),header=None)
            test=pd.read_csv(io.BytesIO(z.read("optdigits.tes")),header=None)
        original=pd.concat([train,test],ignore_index=True)
        assert len(train)==3823 and len(test)==1797
        np.testing.assert_array_equal(frame.to_numpy(),original.iloc[:,:-1].to_numpy())
        np.testing.assert_array_equal(original_target.to_numpy(),original.iloc[:,-1].to_numpy())
        fixed_test[len(train):]=True
        y=(original_target%2==0).astype(int)
        definition="Even digits {0,2,4,6,8} versus odd digits {1,3,5,7,9}"
    else:
        assert set(original_target)=={False,True}
        y=original_target.astype(int)
        definition="Revenue=True versus False"
        frame=frame.drop(columns="PageValues")
    features=frame.columns.tolist()
    frame["binary_target"]=y
    conflicts=frame.groupby(features,dropna=False).binary_target.transform("nunique")>1
    # Preserve official test rows in preference to identical development rows.
    order=np.concatenate([np.flatnonzero(fixed_test),np.flatnonzero(~fixed_test)])
    clean=frame.loc[~conflicts].reindex(order).dropna().drop_duplicates(subset=features).sort_index()
    x=clean[features].copy()
    encoding="Numerical features; selected-training-only standardization"
    if name=="online_shoppers":
        categories={"Month":["Jan","Feb","Mar","Apr","May","June","Jul","Aug","Sep","Oct","Nov","Dec"],
                    "OperatingSystems":list(range(1,9)),"Browser":list(range(1,14)),
                    "Region":list(range(1,10)),"TrafficType":list(range(1,21)),
                    "VisitorType":["New_Visitor","Other","Returning_Visitor"]}
        for column,levels in categories.items():
            assert set(x[column])<=set(levels)
            values=x.pop(column)
            for level in levels:
                x[f"{column}_is_{level}"]=(values==level).astype(float)
        encoding="PageValues excluded before cleaning; fixed categorical indicators; Weekend binary; all scaling uses selected training rows only"
    x=x.astype(float)
    assert np.isfinite(x.to_numpy()).all()
    ids=clean.index.to_numpy()
    np.savez_compressed(folder/"prepared.npz",X=x.to_numpy(),y=clean.binary_target.to_numpy(int),source_ids=ids,is_official_test=fixed_test[ids])
    meta=json.loads(metadata.read_text())["data"]
    provenance=dict(dataset=name,uci_id=spec["uci_id"],source_url=meta["data_url"],doi=meta.get("dataset_doi"),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),license="CC BY 4.0",original_rows=len(frame),
        conflict_rows_removed=int(conflicts.sum()),duplicate_rows_removed=int((~conflicts).sum()-len(clean)),clean_rows=len(clean),
        positive_rows=int(clean.binary_target.sum()),positive_definition=definition,encoded_features=x.shape[1],feature_names=x.columns.tolist(),
        preprocessing=encoding,seed_start=spec["seed"],official_test_rows=int(fixed_test[ids].sum()),
        split_protocol="Official writer-disjoint test; random 75/25 development train/validation" if name=="optical_digits" else "Stratified 60/20/20")
    (folder/"provenance.json").write_text(json.dumps(provenance,indent=2))
    print(name,len(clean),x.shape[1],"positive",int(clean.binary_target.sum()),"conflicts",int(conflicts.sum()),"duplicates",provenance["duplicate_rows_removed"],flush=True)

def main():
    BUNDLE.mkdir(exist_ok=False)
    for name in ("Ada-PU","phase3_confirmatory_S001_S040_100reps","phase5_two_sample_scar_confirmatory_100reps","cluster"):
        shutil.copytree(SOURCE/name,BUNDLE/name,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    code=BUNDLE/"benchmark"
    code.mkdir()
    for name in ("run.py","verify_results.py"):
        shutil.copy2(SOURCE/"benchmark"/name,code/name)
    shutil.copytree(SOURCE/"benchmark/ranking_repair",code/"ranking_repair",ignore=shutil.ignore_patterns("__pycache__"))
    original=(SOURCE/"benchmark/data.py").read_text()
    construction=original[original.index("def construct_data"):]
    construction=construction.replace('    dev, test = train_test_split(indices, test_size=0.2, stratify=y, random_state=seed)',
        '    if os.environ["PU_DATASET"] == "optical_digits":\n        with np.load(ROOT / "data/optical_digits/prepared.npz") as saved:\n            mask = saved["is_official_test"]\n        dev, test = indices[~mask], indices[mask]\n    else:\n        dev, test = train_test_split(indices, test_size=0.2, stratify=y, random_state=seed)')
    (code/"data.py").write_text((ROOT/"data_header.py").read_text()+"\n\n"+construction,encoding="utf-8")
    for name in ("preflight.py",):
        shutil.copy2(ROOT/name,code/name)
    shutil.copy2(ROOT/"cluster.py",BUNDLE/"cluster/cluster.py")
    for name,spec in SPECS.items():
        prepare_dataset(name,spec,code)
    for path in (BUNDLE/"cluster").iterdir():
        if path.suffix in (".sh",".slurm"):
            path.write_text(path.read_text().replace("pu_batch3","pu_batch4"),encoding="utf-8",newline="\n")
    shutil.copy2(ROOT/"README.md",BUNDLE/"README.md")
    (BUNDLE/"preparation").mkdir()
    shutil.copy2(Path(__file__),BUNDLE/"preparation/build_bundle.py")

if __name__=="__main__":
    main()
