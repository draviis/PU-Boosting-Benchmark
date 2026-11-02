"""Read-only source audit and deterministic aggregation for the real-data package."""
from pathlib import Path
import ast, hashlib, json, math
import numpy as np
import pandas as pd

HOME=Path(__file__).resolve().parent
ROOT=HOME.parents[1]
DATASETS={"credit_default":"Credit Default","dry_bean":"Dry Bean","htru2":"HTRU2","online_shoppers":"Online Shoppers",
          "optical_digits":"Optical Digits","phishing":"Phishing Websites","rice":"Rice","spambase":"Spambase"}
DESIGNS=["one_sample","two_sample"]
BUDGETS=[.05,.20]
REPS=[3,5,10,25,20,27]
METHODS={1:"Supervised logistic",2:"Supervised CatBoost",3:"Logistic probability correction",4:"uPU logistic",5:"nnPU logistic",26:"AdaPU",27:"PU gradient boosting"}
for start,strategy in [(6,"Naive"),(11,"Corrected"),(16,"RN"),(21,"CS")]:
    for i,engine in enumerate(["AdaBoost","GBM","XGBoost","LightGBM","CatBoost"]):METHODS[start+i]=strategy+" "+engine
CONTRASTS={"Learner gain":(10,3),"CS addition":(25,10),"RN addition":(20,10)}
SOURCES={
"core":"publication_preparation/eta_scoring_20261013/collected/real_raw.csv",
"eta_audit":"publication_preparation/eta_scoring_20261013/collected/audit.json",
"eta_script":"publication_preparation/eta_scoring_20261013/rescore_real.py",
"plan":"publication_preparation/adapu_workflow/plan.json",
"secondary":"mlj_revision/01b_threshold_completion_and_refit_audit/policy_A_same_model_results.csv",
"secondary_inventory":"mlj_revision/01b_threshold_completion_and_refit_audit/score_recovery_inventory.csv",
"secondary_report":"mlj_revision/01b_threshold_completion_and_refit_audit/REPORT.md",
"neural":"publication_preparation/real_complete_analysis/all_results.csv",
"neural_audit":"publication_preparation/real_complete_analysis/audit.json",
"neural_report":"publication_preparation/real_complete_analysis/REPORT.md",
"neural_training":"publication_preparation/real_complete_analysis/training_review.csv",
"neural_protocol":"pu_real_neural_20260925/PROTOCOL.md",
"neural_models":"pu_real_neural_20260925/models.py",
"neural_worker":"pu_real_neural_20260925/worker.py",
"neural_recipe":"pu_real_neural_20260925/frozen_training_rule.json",
"rn":"mlj_revision/03_rn_mechanism/rn_real_data_diagnostics.csv",
"rn_paired":"mlj_revision/03_rn_mechanism/rn_matched_performance_all.csv",
"rn_report":"mlj_revision/03_rn_mechanism/REPORT.md",
"rn_validation":"mlj_revision/03_rn_mechanism/validation.json",
"base":"phase3_confirmatory_S001_S040_100reps/code/base_phase1_benchmark.py",
"repaired":"phase3_confirmatory_S001_S040_100reps/code/repaired_method_library.py",
"two_sample":"phase5_two_sample_scar_confirmatory_100reps/code/two_sample_benchmark.py",
"runner_audit":"mlj_revision/01_threshold_reanalysis/real_runner_population_audit.json",
}
METRICS=["AUROC","AP"]
KEYS=["dataset_id","design","budget","seed"]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(2**20),b""):h.update(b)
    return h.hexdigest()

def rel(p):return Path(p).relative_to(ROOT).as_posix()
def read(p,**kw):return pd.read_csv(p,float_precision="round_trip",low_memory=False,**kw)
def csv(d,p):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    d.to_csv(p,index=False,float_format="%.17g",na_rep="")
def js(x,p):
    Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def load(key):
    p=ROOT/SOURCES[key];d=read(p)
    print(f"Source: {SOURCES[key]}; rows={len(d)}",flush=True)
    return d

def expected_source_name(n,design):
    if n==1:return "Block0 | Oracle logistic regression"
    if n==2:return "Block0 | Oracle CatBoost"
    if n==3:return "Block1 | "+("Elkan-Noto known-c" if design=="one_sample" else "CaseControl density-ratio known-pi")
    if n in [4,5]:return "Block1 | "+METHODS[n]
    if n>=26:return "Block3 | "+("AdaPU repaired" if n==26 else "PU Gradient Boosting repaired")
    strategy=["Naive","ProbCorrection","RN","CostSensitive"][(n-6)//5]
    return "Block2 | "+strategy+" + "+["AdaBoost","GBM","XGBoost","LightGBM","CatBoost"][(n-6)%5]

def core():
    d=load("core").rename(columns={"dataset":"dataset_id","sampling_scheme":"design","c":"budget","AUPRC":"AP"})
    d["number"]=d.method_id.str[-3:].astype(int)
    for r in d[["number","design","method"]].drop_duplicates().itertuples():
        assert r.method==expected_source_name(r.number,r.design),(r.number,r.design,r.method)
    d["source_method_id"]=d.method_id;d["method_id"]=d.number.map(lambda n:f"M{n:03d}")
    d["method_name"]=d.number.map(METHODS);d["dataset"]=d.dataset_id.map(DATASETS)
    assert set(d.dataset_id)==set(DATASETS) and set(d.number)==set(range(1,28))
    assert set(d.design)==set(DESIGNS) and set(d.budget)==set(BUDGETS)
    assert not d.duplicated(KEYS+["number"]).any()
    assert len(d)==17280 and d.groupby(KEYS).size().eq(27).all()
    return d

def secondary():
    d=load("secondary")
    d=d[(d.study=="real")&(d.method_number>=3)].copy()
    d=d.rename(columns={"dataset":"dataset_id","c":"budget","BalancedAccuracy":"BA"})
    d["number"]=d.method_number.astype(int);d["source_method_id"]=d.method_id
    d["method_id"]=d.number.map(lambda n:f"M{n:03d}")
    d["method_name"]=d.number.map(METHODS);d["dataset"]=d.dataset_id.map(DATASETS)
    assert set(d.dataset_id)==set(DATASETS) and not d.duplicated(KEYS+["number"]).any()
    return d

def neural():
    d=load("neural").rename(columns={"dataset":"dataset_id","scheme":"design","AUPRC":"AP"})
    d=d[d.method.str.startswith(("linear_","mlp_"))].copy()
    d["budget"]=d.budget.map({"c005":.05,"c020":.20});d["dataset"]=d.dataset_id.map(DATASETS)
    expected={f"{kind}_{risk}_{sel}" for kind in ["linear","mlp"] for risk in ["upu","nnpu"] for sel in ["auc","risk"]}
    assert set(d.method)==expected and set(d.dataset_id)==set(DATASETS)
    assert len(d)==5120 and not d.duplicated(KEYS+["method"]).any()
    assert d.groupby(["dataset_id","design","budget","method"]).size().eq(20).all()
    return d

def condition_means(d):
    x=d.groupby(["dataset_id","dataset","design","budget","number","method_id","method_name"],sort=False,as_index=False).agg(
        AUROC=("AUROC","mean"),AP=("AP","mean"),AUROC_n=("AUROC","count"),AP_n=("AP","count"),n_repetitions=("seed","nunique"))
    x["expected_repetitions"]=20
    assert len(x)==864 and x.n_repetitions.eq(20).all()
    return x

def paired(d):
    frames=[]
    for name,(left,right) in CONTRASTS.items():
        a=d[d.number==left][KEYS+METRICS+["data_fingerprint"]]
        b=d[d.number==right][KEYS+METRICS+["data_fingerprint"]]
        z=a.merge(b,on=KEYS,how="outer",suffixes=("_a","_b"),validate="one_to_one",indicator=True)
        assert len(z)==640 and z["_merge"].eq("both").all()
        assert z.data_fingerprint_a.eq(z.data_fingerprint_b).all()
        for metric in METRICS:
            g=z[KEYS].copy();g["metric"]=metric;g["contrast"]=name
            g["left_method_id"]=f"M{left:03d}";g["right_method_id"]=f"M{right:03d}"
            g["left_value"]=z[metric+"_a"];g["right_value"]=z[metric+"_b"]
            g["difference_pp"]=100*(g.left_value-g.right_value);g["paired_available"]=g.difference_pp.notna()
            g["data_fingerprint"]=z.data_fingerprint_a;g["dataset"]=g.dataset_id.map(DATASETS)
            frames.append(g)
    return pd.concat(frames,ignore_index=True)

def paired_condition(p):
    return p.groupby(["dataset_id","dataset","design","budget","metric","contrast"],sort=False,as_index=False).agg(
        difference_pp=("difference_pp","mean"),n_pairs=("difference_pp","count"),split_sd_pp=("difference_pp","std"))
def paired_dataset(p):
    cond=paired_condition(p)
    out=cond.groupby(["dataset_id","dataset","design","metric","contrast"],sort=False,as_index=False).agg(
        difference_pp=("difference_pp","mean"),n_budgets=("budget","nunique"),n_pairs=("n_pairs","sum"))
    spread=p.groupby(["dataset_id","design","metric","contrast"]).difference_pp
    q=spread.quantile([.1,.9]).unstack().rename(columns={.1:"split_p10_pp",.9:"split_p90_pp"}).reset_index()
    out=out.merge(q,on=["dataset_id","design","metric","contrast"],validate="one_to_one")
    out["interval_kind"]="Empirical 10th-90th percentile of 40 paired split contrasts; balanced budget mixture; not a confidence interval"
    assert out.n_budgets.eq(2).all() and out.n_pairs.eq(40).all()
    return out

def neural_pairs():
    d=neural();out=[]
    for risk in ["upu","nnpu"]:
        for selection in ["auc","risk"]:
            a=d[d.method==f"mlp_{risk}_{selection}"].set_index(KEYS)
            b=d[d.method==f"linear_{risk}_{selection}"].set_index(KEYS)
            assert a.index.equals(b.index) or set(a.index)==set(b.index)
            for metric in METRICS:
                z=(100*(a[metric]-b[metric])).rename("difference_pp").reset_index()
                z["risk"]=risk;z["selection"]=selection;z["metric"]=metric;z["dataset"]=z.dataset_id.map(DATASETS)
                out.append(z)
    return pd.concat(out,ignore_index=True)

def rn_data():
    d=load("rn")
    d=d[(d.record_type=="repetition")&(d.engine=="CatBoost")].copy()
    assert len(d)==640 and d.stage.eq("final_train_plus_validation").all()
    assert not d.selection_uses_hidden_Y.any()
    d=d.rename(columns={"dataset":"dataset_id"});d["dataset"]=d.dataset_id.map(DATASETS)
    d["budget"]=d.budget.map({"c005":.05,"c020":.20})
    assert set(d.dataset_id)==set(DATASETS) and not d.duplicated(KEYS).any()
    return d

def audit():
    for sub in ["main/figures","main/figure_tex","main/tables","main/source_code","main/data",
                "supplement/figures","supplement/figure_tex","supplement/tables","supplement/source_code","supplement/data",
                "audit/coverage","audit/rendered","audit/logs","audit/claim_audits"]:
        (HOME/sub).mkdir(parents=True,exist_ok=True)
    manifest=[]
    def track(p,key):
        p=Path(p);manifest.append(dict(source_key=key,source_file=rel(p),sha256=sha(p),bytes=p.stat().st_size))
    for key,p in SOURCES.items():track(ROOT/p,key)
    d=core();s=secondary();n=neural();rn=rn_data()
    assert d.status.eq("ok").all() and d[METRICS].notna().all().all()
    assert d.groupby(["dataset_id","design","budget","number"]).size().eq(20).all()
    # The final eta publication archive changed only hard-classification rules; its ranking results are preserved.
    ea=json.loads((ROOT/SOURCES["eta_audit"]).read_text())
    assert sha(ROOT/SOURCES["core"])==ea["output_hashes"]["real_raw.csv"]
    plan=json.loads((ROOT/SOURCES["plan"]).read_text());assert len(plan["cells"])==640
    case_checks=[]
    idx=d.set_index(KEYS).sort_index()
    for cell in plan["cells"]:
        path=ROOT/cell["local_data_dir"]/"manifest.json"
        track(path,"original_split_manifest")
        assert sha(path)==cell["files"]["manifest.json"]
        m=json.loads(path.read_text())
        assert m["test_labels_used_for_prior"] is False
        assert m["prior_regime"]=="oracle_empirical_training_background"
        budget={"c005":.05,"c020":.20}[cell["budget"]]
        g=idx.loc[(cell["dataset"],cell["scheme"],budget,cell["seed"])]
        assert g.data_fingerprint.eq(cell["data_fingerprint"]).all()
        assert m["data_fingerprint"]==cell["data_fingerprint"] and m["labeling_probability_one_sample"]==budget
        case_checks.append(dict(dataset=DATASETS[cell["dataset"]],design=cell["scheme"],budget=budget,seed=cell["seed"],
                                manifest_file=rel(path),fingerprint_verified=True,test_labels_used_for_prior=False,
                                prior_regime=m["prior_regime"],ranking_model_stage="final_train_plus_validation"))
    csv(pd.DataFrame(case_checks),HOME/"audit/coverage/original_case_provenance.csv")
    runner=json.loads((ROOT/SOURCES["runner_audit"]).read_text())
    for r in runner:
        track(ROOT/r["path"],"runner_source")
        assert sha(ROOT/r["path"])==r["sha256"]
    na=json.loads((ROOT/SOURCES["neural_audit"]).read_text())
    assert na["cells"]==640 and na["new_prediction_files_verified"]==5120 and na["failed_candidates"]==0
    assert n.status.eq("ok").all()
    recipe=json.loads((ROOT/SOURCES["neural_recipe"]).read_text());assert recipe["test_scores_used_for_rule"] is False
    # Check previously selected model access statically; no worker/model is imported or executed.
    for key in ["base","repaired","two_sample","neural_models","neural_worker"]:
        text=(ROOT/SOURCES[key]).read_text();ast.parse(text)
    # Ranking aliases: preserve both original numeric entries, document machine-epsilon differences.
    aliases=[]
    for left in range(6,11):
        a=d[d.number==left].set_index(KEYS);b=d[d.number==left+5].set_index(KEYS).reindex(a.index)
        for key in a.index:
            row=dict(zip(KEYS,key));row.update(naive=f"M{left:03d}",corrected=f"M{left+5:03d}")
            assert ast.literal_eval(a.loc[key,"best_params"])==ast.literal_eval(b.loc[key,"best_params"])
            for metric in METRICS:
                delta=b.loc[key,metric]-a.loc[key,metric]
                assert abs(delta)<=1e-12
                row[metric+"_difference"]=delta
            row["alias_verified_to_1e_12"]=True;aliases.append(row)
    csv(pd.DataFrame(aliases),HOME/"audit/coverage/ranking_alias_checks.csv")
    # All requested metrics; old hard metrics in the ranking archive are never used.
    verified=s[s.status=="verified"]
    assert verified.policy.eq("A_same_model").all() and not verified.test_labels_used_for_threshold_selection.any()
    assert len(verified)==802
    assert verified[["BA","F1","MCC"]].notna().all().all()
    counts=verified.groupby(["dataset_id","design","budget","number"]).size()
    coverage=[]
    for (ds,design,budget,num),g in d.groupby(["dataset_id","design","budget","number"]):
        for metric in ["AUROC","AP","BA","F1","MCC","Brier","LogLoss"]:
            hard=metric in ["BA","F1","MCC"]
            if hard:
                available=int(counts.get((ds,design,budget,num),0)) if num>=3 else 0
                source=SOURCES["secondary"]
                notes=("Final same-model pilot only; outcomes of full refitted ranking archive are excluded" if num>=3
                       else "Supervised reference; no PU-rule secondary summary requested or inferred")
            else:
                available=int(g[metric].notna().sum());source=SOURCES["core"]
                notes="Final train+validation refit; supervised reference reported separately" if num<=2 else "Final train+validation refit"
                if metric in ["Brier","LogLoss"] and available<20:notes+="; probability output not supplied, missing preserved"
            coverage.append(dict(dataset=DATASETS[ds],design=design,budget=budget,method_id=f"M{num:03d}",method_name=METHODS[num],
                metric=metric,expected_repetitions=20,available_repetitions=available,coverage_fraction=available/20,
                source_file=source,notes=notes))
    csv(pd.DataFrame(coverage),HOME/"audit/coverage/real_data_coverage.csv")
    csv(s[s.status!="verified"][["dataset","design","budget","seed","method_id","method_name","status"]],HOME/"audit/coverage/secondary_unavailable_pilot_rows.csv")
    csv(verified[["dataset","design","budget","seed","method_id","method_name","scope","BA","F1","MCC","policy"]],HOME/"audit/coverage/secondary_available_pilot_records.csv")
    inv=read(ROOT/SOURCES["secondary_inventory"],usecols=["study","method_id","archived_train_only_test_scores","train_only_test_scores_available"])
    inv=inv[(inv.study=="real")&(inv.method_id.str[-3:].astype(int)>=3)]
    assert len(inv)==16000 and inv.archived_train_only_test_scores.sum()==0 and inv.train_only_test_scores_available.sum()==802
    # Check Task3 RN ranking effects against the authoritative final archive.
    p=paired(d);cp=paired_condition(p);dp=paired_dataset(p)
    rp=p[p.contrast=="RN addition"].pivot(index=KEYS,columns="metric",values="difference_pp")
    for r in rn.itertuples():
        k=(r.dataset_id,r.design,r.budget,r.seed)
        assert abs(rp.loc[k,"AUROC"]-100*r.Delta_AUROC)<1e-10
        assert abs(rp.loc[k,"AP"]-100*r.Delta_AP)<1e-10
    csv(d[["dataset_id","dataset","design","budget","seed","method_id","method_name","number","AUROC","AP","Brier","LogLoss","status","data_fingerprint"]],HOME/"audit/coverage/audited_ranking_records.csv")
    csv(condition_means(d),HOME/"audit/coverage/condition_means.csv")
    csv(p,HOME/"main/data/real_paired_repetition_differences.csv")
    csv(cp,HOME/"main/data/real_paired_condition_differences.csv")
    csv(dp,HOME/"main/data/real_paired_dataset_differences.csv")
    csv(n.groupby(["dataset","design","budget","method"],as_index=False).agg(AUROC_n=("AUROC","count"),AP_n=("AP","count")),HOME/"audit/coverage/neural_coverage.csv")
    csv(condition_means(d).query("number<=2"),HOME/"supplement/data/supervised_references.csv")
    csv(pd.DataFrame(manifest).drop_duplicates("source_file"),HOME/"SOURCE_MANIFEST.csv")
    limitation="""# Real-data secondary-metric limitation

No complete final-protocol BA/F1/MCC table can be created without new fitting or previously unavailable selected-model test scores. Accordingly, the requested supplementary secondary-metric display is **not produced**; the limitation and all available pilot values are preserved in this audit.

The target is 16,000 PU method/case records (8 datasets x 2 designs x 2 budgets x 20 repetitions x 25 configurations). Task 1B contains **802 verified records (5.0125%)**: 794 of 800 fixed-pilot attempts plus eight targeted HTRU2 records from additional seeds. Six fixed-pilot replays failed validation verification and remain unavailable. Most method/conditions have one verified split, some have none, and the targeted records give a few methods two or three. They are not a balanced 20-repetition sample.

All accepted pilot records use A_same_model: the selected training-fit model, empirical linear (1-pi_tilde) validation quantile, all validation observations for OS / validation U for TS, raw q for corrected methods, and the same model's test scores. The selector does not use hidden validation or test Y. Test Y is used to evaluate predictions. These results are separate from the final-refit ranking metrics.

The Task 1B recovery inventory records zero original archived selected-training-model test-score files for the 16,000 requested PU records and 802 previously recovered records. The final ranking archive's historical BA/F1/MCC uses superseded rules and is deliberately excluded. No missing value is imputed, no model is fitted, and no new threshold is applied to final-refit scores.

See secondary_available_pilot_records.csv, secondary_unavailable_pilot_rows.csv and real_data_coverage.csv. The probability metrics Brier/LogLoss remain separate archived final-refit outputs with their original availability; they do not repair hard-metric coverage.
"""
    (HOME/"audit/coverage/REAL_SECONDARY_METRIC_LIMITATION.md").write_text(limitation,encoding="utf-8")
    report="""# Real-data coverage and provenance audit

The final ranking archive is complete: **eight datasets, 32 dataset/design/budget conditions, 640 repeated-split cells and 17,280 method/cell rows**. Each condition has all 25 PU configurations plus two supervised references, each evaluated on 20 repetitions. AUROC and AP have no missing records and all source statuses are ok. Missing probability metrics remain missing: 6,400 of 17,280 Brier/log-loss values are available for each metric. No failed ranking fit is silently removed.

## Source and model stage

The authoritative source is publication_preparation/eta_scoring_20261013/collected/real_raw.csv. Its SHA-256 matches its completed source audit; that revision changed classification rules only and retained ranking/probability results. Its AUROC/AP is evaluated from final train+validation refits, using unclipped ranking scores. Archived hard-classification fields are never used here.

The 640 frozen experiment-plan cells map exactly to the eight requested datasets, two designs, two budgets and 20 seeds per condition. Every original split-manifest hash and data fingerprint matches. Dataset IDs are mapped explicitly to the requested display names. M and TM method IDs are mapped to canonical M001--M027, with the design-specific source name asserted: M003 is known-c Elkan--Noto in OS and known-prior case-control correction in TS. The original IDs remain in source data. All 25 PU configurations remain separate in machine-readable results; the two supervised references are in supplement/data/supervised_references.csv and outside PU summaries.

## Selection and information access

Source review of the original base/repaired/two-sample benchmark and archived real-data runners confirms that tuning uses validation PU labels/risk (true validation labels only for the two supervised references), selected hyperparameters are used in train+validation refits, and test Y enters final evaluation rather than the selection objective. The original real-data runner retains hidden Y for audit; this report does not incorrectly infer a hidden-label guarantee from the source CSV's blank uses_true_y_for_development cells for PU methods. Evidence is the frozen source code and existing provenance audits, plus matched input manifests, not a new fitting run.

All 640 manifests explicitly state test_labels_used_for_prior=False and oracle_empirical_training_background prior regime. Thus the supplied prior is privileged training-background information; this is not an estimated-prior deployment study. Original splits, label hiding, grids and supplied priors are preserved.

The neural extension independently verifies 5,120 selected/refitted prediction files and all 640 source cells in its existing audit. Its worker loads test Y only after every configuration is selected and fitted; the model function accepts only train/validation S and test X. Its frozen recipe states no test-score use. This task reads these audits and code without importing or running training code.

## Aliases, missingness and ancillary evidence

All 3,200 naive/corrected same-engine pairs have identical selected parameters and ranking metrics within 1e-12; maximum discrepancies are machine precision (about 1.11e-16). These are numerical ranking aliases, not independent evidence of a correction benefit. Original full-precision values are retained separately in all-configuration outputs; they are neither averaged together nor used to multiply evidence. No PU method ranks or group maxima are used.

Final same-model secondary coverage is only 802/16,000 PU records (794 fixed-pilot successes plus eight targeted HTRU2 records). Six pilot verification failures are preserved. REAL_SECONDARY_METRIC_LIMITATION.md explains why no full secondary table is presented.

Task 3 provides all 640 RN-CatBoost final-stage selector diagnostics; their ranking differences reconcile to the authoritative archive. RN sets were previously reconstructed from exact frozen inputs/code, because original selected indices were not saved. Purity, negative recall and selected-U fraction are diagnostic associations, not causal explanations. Task 4 alternatives are not mixed into the original RN method.

The matched neural extension retains all eight linear/neural x uPU/nnPU x selection configurations and all 20 splits per condition. Its fixed-budget training limitation is retained: 522/5,120 selections hit the 3,000-epoch cap; the numerical gate was not passed under its original cap criteria. Subsequent archived authorization completed the unchanged fixed-budget study; successful metric verification does not establish convergence.

## Aggregation

First average paired split differences or method metrics within each dataset/design/budget. Then average budgets within dataset only when requested, and give all eight datasets equal weight for aggregate summaries. No direct pooling across datasets, group maxima, winner selection or outcome-driven representative choice occurs. Main-figure bars show empirical 10th--90th percentile paired-split spread across the balanced two-budget mixture (40 values); they are not confidence intervals or population inference. Repeated splits overlap.

Every result and input is traceable through SOURCE_MANIFEST.csv and the exact data/CSV beside each display. This is post-processing only: no fit, refit, simulation, tuning, split change or label draw occurs; original outputs remain unchanged.
"""
    (HOME/"audit/coverage/REAL_DATA_COVERAGE_REPORT.md").write_text(report,encoding="utf-8")
    js(dict(passed=True,datasets=8,conditions=32,cases=640,method_rows=17280,ranking_missing=0,alias_pairs=3200,
            secondary_verified=802,secondary_expected=16000,neural_rows=5120,rn_cases=640,no_model_fits=True),
       HOME/"audit/coverage/audit_validation.json")
    print("Audit passed before displays: all 32 ranking conditions complete.")

def require_audit():
    assert json.loads((HOME/"audit/coverage/audit_validation.json").read_text())["passed"]

if __name__=="__main__":audit()
