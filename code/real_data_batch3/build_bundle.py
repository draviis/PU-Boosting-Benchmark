"""Freeze a three-dataset study using the established benchmark methods."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
BUNDLE = ROOT / "batch3_math_20260914"
SOURCE = REPO / "real_data_htru2/htru2_math_20reps_20260914"
SPECS = {
    "rice": dict(uci_id=545, n=3810, target="Class", positive="Cammeo", seed=918000),
    "phishing": dict(uci_id=327, n=11055, target="result", positive=-1, seed=920000),
    "credit_default": dict(uci_id=350, n=30000, target="Y", positive=1, seed=922000),
}

def prepare_dataset(name, spec, code):
    folder = code / "data" / name
    folder.mkdir(parents=True)
    csv = ROOT / "data" / f"{spec['uci_id']}.csv"
    metadata_file = ROOT / "data" / f"{spec['uci_id']}_metadata.json"
    shutil.copy2(csv, folder / "source.csv")
    shutil.copy2(metadata_file, folder / "uci_metadata.json")
    metadata = json.loads(metadata_file.read_text())["data"]
    frame = pd.read_csv(csv)
    assert len(frame) == spec["n"] and not frame.isna().any().any()
    target = spec["target"]
    assert frame[target].nunique() == 2
    features = [c for c in frame if c not in (target, "ID")]
    conflicts = frame.groupby(features, dropna=False)[target].transform("nunique") > 1
    cleaned = frame.loc[~conflicts].drop_duplicates(subset=features)
    x = cleaned[features].astype(float)
    encoding = "Original numerical/discrete feature codes, standardized on selected training rows"
    if name == "credit_default":
        # Fixed indicator schema: no category discovery or fitted transform
        # uses validation/test rows. Retain undocumented codes explicitly.
        categories = {"X2": [1,2], "X3": [0,1,2,3,4,5,6], "X4": [0,1,2,3]}
        for column, levels in categories.items():
            assert set(x[column]) <= set(levels)
            values = x.pop(column)
            for level in levels:
                x[f"{column}_is_{level}"] = (values == level).astype(float)
        encoding = "ID excluded; fixed indicators for sex, education and marriage (including undocumented codes); remaining 20 variables numeric; train-only scaling"
    y = (cleaned[target] == spec["positive"]).to_numpy(int)
    assert np.isfinite(x.to_numpy()).all() and 0 < y.sum() < len(y)
    np.savez_compressed(folder / "prepared.npz", X=x.to_numpy(float), y=y, source_ids=cleaned.index.to_numpy())
    provenance = dict(dataset=name, source_url=metadata["data_url"], uci_id=spec["uci_id"],
        doi=metadata.get("dataset_doi"), source_sha256=hashlib.sha256(csv.read_bytes()).hexdigest(),
        original_rows=len(frame), conflict_rows_removed=int(conflicts.sum()),
        duplicate_rows_removed=int((~conflicts).sum()-len(cleaned)), clean_rows=len(cleaned),
        positive_definition=f"{target}={spec['positive']}", positive_rows=int(y.sum()),
        original_features=len(features), encoded_features=x.shape[1], feature_names=x.columns.tolist(),
        preprocessing=encoding, seed_start=spec["seed"], license="CC BY 4.0")
    (folder / "provenance.json").write_text(json.dumps(provenance,indent=2))
    print(name, len(cleaned), x.shape[1], int(y.sum()), "conflicts",int(conflicts.sum()),flush=True)

def main():
    BUNDLE.mkdir(exist_ok=False)
    for name in ("Ada-PU", "phase3_confirmatory_S001_S040_100reps", "phase5_two_sample_scar_confirmatory_100reps", "cluster"):
        shutil.copytree(SOURCE/name,BUNDLE/name,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    code = BUNDLE / "benchmark"
    code.mkdir()
    for name in ("run.py","verify_results.py"):
        text = (SOURCE/"real_data_htru2"/name).read_text().replace("HTRU2","the selected dataset").replace("real_data_htru2","benchmark")
        (code/name).write_text(text,encoding="utf-8")
    shutil.copytree(SOURCE/"real_data_htru2/ranking_repair",code/"ranking_repair",ignore=shutil.ignore_patterns("__pycache__"))
    original = (SOURCE/"real_data_htru2/data.py").read_text()
    construction = original[original.index("def construct_data"):].replace("load_htru2()","load_dataset()")
    construction = construction.replace('"dataset": "HTRU2"', '"dataset": os.environ["PU_DATASET"]')
    (code/"data.py").write_text((ROOT/"data_header.py").read_text()+"\n\n"+construction,encoding="utf-8")
    shutil.copy2(ROOT/"preflight.py",code/"preflight.py")
    shutil.copy2(ROOT/"cluster.py",BUNDLE/"cluster/cluster.py")
    for name,spec in SPECS.items():
        prepare_dataset(name,spec,code)
    for path in (BUNDLE/"cluster").iterdir():
        if path.suffix not in (".slurm",".sh"):
            continue
        text = path.read_text().replace("htru2", "pu_batch3")
        text = text.replace("1-80%10","1-240%10").replace("1-4%4","1-4,81-84,161-164%4")
        text = text.replace("5-80%10","5-80,85-160,165-240%10")
        text = text.replace("04:00:00","08:00:00").replace("00:30:00","01:00:00")
        path.write_text(text,encoding="utf-8",newline="\n")

if __name__ == "__main__":
    main()
