"""Fixed-method analyses for the complete approved nonlinear extension."""
from pathlib import Path
import json,math,hashlib
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parent
METRICS=['AUROC','AP','BA','F1','MCC']
CONTRASTS=[(f'M{n+10:03d}',f'M{n:03d}','RN_minus_naive') for n in range(6,11)]+[(f'M{n+15:03d}',f'M{n:03d}','CS_minus_naive') for n in range(6,11)]+[(f'M{a:03d}',f'M{b:03d}','fixed_nonboosting_minus_fixed_boosting') for a in [3,4,5] for b in [10,25,26,27]]
def save(name,obj):
    with (H/name).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
def records(frame):return json.loads(frame.to_json(orient='records',double_precision=15))
def summarize(group):
    result={'n_repetitions':len(group)}
    for metric in METRICS:
        v=group[metric].dropna().to_numpy();assert len(v)>1
        se=float(np.std(v,ddof=1)/np.sqrt(len(v)))
        result[metric+'_n']=int(len(v));result[metric+'_missing']=int(len(group)-len(v))
        result.update({metric+'_mean':float(v.mean()),metric+'_MCSE':se,metric+'_MC_lower':float(v.mean()-1.96*se),metric+'_MC_upper':float(v.mean()+1.96*se)})
    return result
def main():
    validation=json.loads((H/'main_validation.json').read_text());assert validation['passed'] and validation['method_rows']==32400
    rows=json.loads((H/'nonlinear_results.json').read_text())
    for row in rows:
        row['refit_available']=row.get('refit_available',True)
        row['refit_failure_reason']=row.get('refit_failure_reason') or ''
    d=pd.DataFrame(rows)
    assert int((~d.refit_available).sum())==6
    assert d[['BA','F1','MCC']].notna().all().all()
    assert (d[['AUROC','AP']].notna().all(axis=1)==d.refit_available).all()
    assert len(d)==32400 and len(d[d.supervised_reference])==2400
    assert not d.duplicated(['scenario_id','design','rep','method_id']).any()
    assert set(d.rep)==set(range(100)) and (d.groupby(['scenario_id','design','method_id']).size()==100).all()
    assert d.hard_metric_model.eq('selected_train_only').all() and not d.threshold_transfer_used.any()
    calib={r['scenario_id']:r for r in json.loads((H/'nonlinear_dgp_calibration.json').read_text())}
    summary=[]
    for (sc,des,mid),g in d.groupby(['scenario_id','design','method_id'],sort=True):
        z=dict(scenario_id=sc,design=des,method_id=mid,method=g.method.iloc[0],family=g.family.iloc[0],supervised_reference=bool(g.supervised_reference.iloc[0]),**summarize(g))
        z.update(Bayes_AUROC=calib[sc]['bayes_auroc'],Bayes_BA_unconstrained=calib[sc]['bayes_ba'])
        z.update(AUROC_regret=z['Bayes_AUROC']-z['AUROC_mean'],BA_gap_to_unconstrained_Bayes=z['Bayes_BA_unconstrained']-z['BA_mean'])
        summary.append(z)
    summary=pd.DataFrame(summary)
    aggregates=[]
    for (des,mid),g in d.groupby(['design','method_id'],sort=True):
        for rule in ['simple_six_DGP_mean','equal_five_mechanism_mean']:
            weights={f'NL{i:02d}':1/6 for i in range(1,7)} if rule=='simple_six_DGP_mean' else {'NL01':.1,'NL05':.1,'NL02':.2,'NL03':.2,'NL04':.2,'NL06':.2}
            byscenario=summary[(summary.design==des)&(summary.method_id==mid)].set_index('scenario_id')
            z=dict(design=des,method_id=mid,method=g.method.iloc[0],supervised_reference=bool(g.supervised_reference.iloc[0]),aggregation=rule,n_repetitions=100,refit_missing=int((~g.refit_available).sum()))
            for metric in METRICS:
                mean=sum(w*float(byscenario.loc[sc,metric+'_mean']) for sc,w in weights.items())
                se=math.sqrt(sum(w*w*float(byscenario.loc[sc,metric+'_MCSE'])**2 for sc,w in weights.items()))
                z.update({metric+'_mean':mean,metric+'_MCSE':se,metric+'_MC_lower':mean-1.96*se,metric+'_MC_upper':mean+1.96*se,metric+'_n_available':int(byscenario[metric+'_n'].sum()),metric+'_n_expected':600,metric+'_min_scenario_n':int(byscenario[metric+'_n'].min())})
            aggregates.append(z)
    aggregates=pd.DataFrame(aggregates)
    paired=[]
    for (sc,des),g in d.groupby(['scenario_id','design']):
        for a,b,kind in CONTRASTS:
            ga=g[g.method_id.eq(a)].set_index('rep');gb=g[g.method_id.eq(b)].set_index('rep')
            assert set(ga.index)==set(gb.index)==set(range(100))
            for metric in METRICS:
                diff=(ga[metric]-gb[metric]).sort_index().dropna();assert len(diff)>1
                se=float(diff.std(ddof=1)/np.sqrt(len(diff)));mean=float(diff.mean())
                paired.append(dict(scenario_id=sc,design=des,method_A=a,method_B=b,contrast_type=kind,metric=metric,n_pairs=int(len(diff)),expected_pairs=100,missing_pairs=int(100-len(diff)),conditional_on_joint_fit_availability=bool(len(diff)<100),mean_difference=mean,MCSE=se,MC_lower=mean-1.96*se,MC_upper=mean+1.96*se))
    ranks=[]
    for (sc,des),g in summary[~summary.supervised_reference].groupby(['scenario_id','design']):
        z=g[['scenario_id','design','method_id','method','family','AUROC_n','AP_n','BA_n']].copy()
        for metric in ['AUROC','AP','BA']:z[metric+'_rank']=g[metric+'_mean'].rank(ascending=False,method='average')
        ranks.extend(records(z))
    diagnostics=[]
    for (sc,des,mid),g in d.groupby(['scenario_id','design','method_id']):
        diagnostics.append(dict(scenario_id=sc,design=des,method_id=mid,method=g.method.iloc[0],n=100,validation_population=g.validation_population.iloc[0],validation_n=int(g.validation_n.iloc[0]),mean_validation_positive_fraction=float(g.validation_predicted_positive_fraction.mean()),mean_test_positive_fraction=float(g.test_predicted_positive_fraction.mean()),validation_fraction_material_deviation_cases=int(g.validation_material_prevalence_deviation.sum()),test_fraction_material_deviation_cases=int(g.test_material_prevalence_deviation.sum()),cases_with_validation_threshold_ties=int((g.validation_ties_at_threshold>0).sum()),mean_validation_max_tie_mass=float(g.validation_max_tie_mass.mean()),mean_test_max_tie_mass=float(g.test_max_tie_mass.mean()),material_definition='absolute predicted-positive-fraction difference from .20 exceeds .02'))
    aliases=validation['matched_naive_corrected'];assert len(aliases)==6000
    tables={
      'nonlinear_results.csv':rows,
      'nonlinear_scenario_summary.csv':records(summary),
      'nonlinear_aggregate_summary.csv':records(aggregates),
      'nonlinear_paired_mc_uncertainty.csv':paired,
      'nonlinear_fixed_method_ranks.csv':ranks,
      'nonlinear_threshold_diagnostics.csv':diagnostics,
      'matched_naive_corrected_checks.csv':aliases,
      'nonlinear_supervised_references.csv':records(d[d.supervised_reference])}
    save('analysis_tables.json',tables)
    save('analysis_validation.json',dict(passed=True,complete_cases=1200,method_rows=32400,PU_rows=30000,supervised_rows=2400,scenario_method_cells=len(summary),aggregate_cells=len(aggregates),paired_contrasts=len(paired),alias_pairs_equal=len(aliases),refit_metrics_available=int(d.refit_available.sum()),refit_metrics_unavailable=int((~d.refit_available).sum()),hard_metrics_available=len(d),all_DGPs_individual_primary=True,all_analyses_fixed_method=True,planned_methods=27,expected_repetitions_per_cell=100))
    report(summary,aggregates,pd.DataFrame(paired),pd.DataFrame(diagnostics),calib,d)
    print(json.dumps({k:len(v) for k,v in tables.items()}))
def report(summary,aggregates,paired,diag,calib,d):
    out=['# Nonlinear simulation extension: fixed methods and refit availability','',
    'The approved six DGPs were run under both sampling designs with 100 repetitions each. All 32,400 method records are accounted for: 30,000 practical-PU rows and 2,400 supervised-reference rows. BA/F1/MCC are available for all 32,400; final-refit AUROC/AP are available for 32,394. Six RN refits in four cases had no negatives and are explicitly unavailable, with no imputation or substitute model. Thus 1,196 cases have every requested metric and four have partial ranking coverage. Original S001--S040, the nonlinear parameters and their calibration are preserved. No Task 9 methods, BaggingPU, or nonlinear-spy RN variants are included.',
    '',
    '## How to read the metrics','',
    'AUROC and AP use final train+validation refits. BA, F1 and MCC are secondary metrics from the selected train-only model. The empirical .80 validation quantile is applied to test scores from that same model: all 5,000 validation observations in one-sample and the 4,700 validation U observations in two-sample. Corrected methods use raw q. No threshold transfer or mixed original rules are used. The two supervised references follow the same secondary threshold diagnostic, but use their original supervised training/selection and are excluded from practical PU ranks.',
    '',
    'Selected validation/test score arrays are saved for all method records. Both refitted arrays are also saved for all 32,394 successful refits; the six unavailable refits have no invented score arrays. Masks, evaluation labels, candidates and selected parameters are saved on Augsburg under this Task 7 directory. Model-facing hidden labels are redacted for practical PU fitting. Test labels are used only after fitting/selection for evaluation. The collector independently recomputed every reported metric and all 6,000 naive/corrected paired checks. The pairs have identical raw scores, hard predictions and BA.',
    '',
    'The historical NONLINEAR_DGP_PLAN.md is retained unchanged to preserve the pre-calibration record; its old catalog and threshold-transfer passages are superseded by EXECUTION_PROTOCOL.md and execution_manifest.json. Calibration was not repeated.',
    '',
    '## Individual DGP results are primary','',
    'The table below shows predeclared fixed comparators, not group maxima. Each entry is AUROC / AP / BA, averaged over 100 repetitions where available. Six RN scenario/method cells have 99 valid refits; their ranking means are conditional on refit success. The CSVs give metric-specific counts. All 25 PU methods and both supervised references appear in nonlinear_scenario_summary.csv and the comparison PDF.']
    ids=['M003','M004','M005','M010','M020','M025','M026','M027']
    names={'M003':'Corrected logistic','M004':'uPU logistic','M005':'nnPU logistic','M010':'Naive CatBoost','M020':'RN CatBoost','M025':'CS CatBoost','M026':'AdaPU','M027':'PU gradient boosting'}
    for des in ['one_sample','two_sample']:
        out+=['',f'### {des.replace("_"," ")}','', '| Fixed method | NL01 | NL02 | NL03 | NL04 | NL05 | NL06 |','|---|---|---|---|---|---|---|']
        for mid in ids:
            g=summary[(summary.design==des)&(summary.method_id==mid)].set_index('scenario_id')
            out.append('| '+names[mid]+' | '+' | '.join(f"{g.loc[sc,'AUROC_mean']:.3f} / {g.loc[sc,'AP_mean']:.3f} / {g.loc[sc,'BA_mean']:.3f}" for sc in [f'NL{i:02d}' for i in range(1,7)])+' |')
    out+=['','## Fixed paired differences and Monte Carlo uncertainty','',
    'Every contrast uses repetition-level differences on the same data split. MCSE is sd(difference)/sqrt(n_pairs), with n_pairs=100 for complete contrasts and 99 for the affected RN ranking contrasts; intervals are mean +/- 1.96 MCSE. Missing ranking pairs are not imputed. They quantify finite simulation precision and are descriptive, without multiple-comparison claims. The six designed DGPs are not treated as random observations for hypothesis tests. All RN-minus-naive and CS-minus-naive engine pairs, plus each fixed nonboosting baseline against four fixed boosting comparators, are in nonlinear_paired_mc_uncertainty.csv.',
    '', '| Design | DGP | Contrast | AUROC difference [MC interval] | BA difference [MC interval] |','|---|---|---|---|---|']
    for des in ['one_sample','two_sample']:
        for sc in ['NL01','NL02','NL03','NL04','NL05','NL06']:
            for a,b,label in [('M025','M010','CS - naive CatBoost'),('M005','M027','nnPU logistic - PU GB')]:
                # The second contrast is predeclared among fixed nonboosting/boosting pairs.
                g=paired[(paired.design==des)&(paired.scenario_id==sc)&(paired.method_A==a)&(paired.method_B==b)].set_index('metric')
                out.append('| '+des+' | '+sc+' | '+label+' | '+' | '.join(f"{g.loc[m,'mean_difference']:+.3f} [{g.loc[m,'MC_lower']:+.3f}, {g.loc[m,'MC_upper']:+.3f}]" for m in ['AUROC','BA'])+' |')
    out+=['','## Aggregate sensitivity','',
    'Both requested aggregates are in nonlinear_aggregate_summary.csv. The simple mean assigns each DGP weight 1/6. The equal-five-mechanism mean assigns NL01 and NL05 weight 1/10 each, and NL02, NL03, NL04 and NL06 weight 1/5 each. Thus the sinusoidal mechanism receives 20% rather than 33.3%. Individual DGP results remain primary. Each aggregate first averages available repetitions within each DGP, then applies the exact declared DGP weights. Its MCSE is sqrt(sum(weight^2 * DGP_MCSE^2)), using independence of the DGP streams. Missing refits therefore do not change DGP weights. Affected RN aggregates are conditional on successful refitting; this is simulation uncertainty, not between-scenario hypothesis testing.',
    '', '| Design | Fixed method | Six-DGP AUROC / AP / BA | Five-mechanism AUROC / AP / BA |','|---|---|---|---|']
    for des in ['one_sample','two_sample']:
        for mid in ids:
            g=aggregates[(aggregates.design==des)&(aggregates.method_id==mid)].set_index('aggregation')
            out.append('| '+des+' | '+names[mid]+' | '+' | '.join(' / '.join(f"{g.loc[rule,m+'_mean']:.3f}" for m in ['AUROC','AP','BA']) for rule in ['simple_six_DGP_mean','equal_five_mechanism_mean'])+' |')
    out+=['','## Bayes information and threshold ties','',
    'The exact likelihood-ratio calibration is unchanged. AUROC regret is relative to the independent million-observation Bayes calibration for each DGP. The Bayes BA ceiling uses likelihood ratio >= 1 (equal class weighting); it is unconstrained and therefore need not be attained by a prevalence-quantile rule. Calibration Monte Carlo uncertainty is reported separately in nonlinear_dgp_calibration.csv, rather than being silently added to repetition MCSE.',
    '', '| DGP | Frozen Bayes AUROC | Frozen Bayes BA |','|---|---|---|']
    for sc,c in calib.items():out.append(f"| {sc} | {c['bayes_auroc']:.6f} | {c['bayes_ba']:.6f} |")
    out+=['',f"Across all method rows, {int(diag.validation_fraction_material_deviation_cases.sum())} validation and {int(diag.test_fraction_material_deviation_cases.sum())} test predicted-positive fractions differ from .20 by more than .02. Score ties are retained with the declared >= comparison, so the positive fraction can exceed the target. These counts include shared-score correction aliases and supervised references; use nonlinear_threshold_diagnostics.csv for method-specific interpretation.",
    '',
    '## Files and scope','',
    '- nonlinear_results.csv: 32,400 repetition-level rows, with model-stage and threshold diagnostics.',
    '- nonlinear_scenario_summary.csv: all 324 DGP/design/method cells with Monte Carlo uncertainty and Bayes gaps.',
    '- nonlinear_aggregate_summary.csv: both requested weighting schemes.',
    '- nonlinear_paired_mc_uncertainty.csv: fixed paired contrasts.',
    '- nonlinear_fixed_method_ranks.csv: within-DGP ranks of the 25 PU methods only; tied ranks are averaged.',
    '- nonlinear_supervised_references.csv: separate supervised repetition rows.',
    '- nonlinear_fixed_method_comparison.pdf: fixed methods, with supervised rows separated.',
    '- matched_naive_corrected_checks.csv and nonlinear_threshold_diagnostics.csv: threshold audits.',
    '',
    'This is a separate nonlinear extension, not a replacement or renumbering of the original 40 scenarios. Its supplied-prior information condition does not resolve the estimated-prior limitations found in Task 6. No manuscript files were edited.']
    unavailable=d[~d.refit_available]
    out+=['','## Execution repairs and remaining unavailable refits','',
    'The preflight passed all 108 method rows. A line-ending checksum mismatch initially prevented dispatch; byte-preserving transfer repaired it. The first main run saved 32,244 rows and left 78 RN cells failed, which also blocked 78 subsequent CS fits. Completed output files were preserved.',
    '',
    'RN recovery skips only one-class training candidates that the affected engine cannot fit, retaining the original selector, grid values and validation criterion. Six regression checks and independent four-engine pilot parity passed. Seventy-two RN results and all 78 blocked CS results were recovered. This is an explicit error-handling exception to the original tuning control flow, not a new RN selector. Existing successful LightGBM cells were preserved, including their engine-specific handling of degenerate candidates.',
    '',
    'For the remaining six RN cells, the selected train-only model is valid but the selected final refit has zero negatives. Their selected-model scores and hard metrics are retained. AUROC/AP are blank, not zero or chance values. No parameter reselection, constant-positive fallback, or borrowing of selected-model AUROC/AP occurred. Missingness can depend on the data, so the 99-refit RN means should be read alongside their failure rate rather than as unconditional performance.',
    '', '| Case | Method | Refit ranking status | Selected hard metrics |','|---|---|---|---|']
    for _,r in unavailable.iterrows():out.append(f"| {r['case']} | {r['method_id']} | unavailable: no RN | retained |")
    out+=['','See EXECUTION_REPAIRS_20261018.md, rn_recovery_protocol.json, rn_empty_refit_protocol.json and the recovery validation files for the full audit. The analysis ZIP includes those records. No original S001--S040 results or manuscript files were changed.']
    # The status/plan-stage report is already archived; only the current status report is updated.
    (H/'REPORT.md').write_text('\n'.join(out)+'\n',encoding='utf-8')
if __name__=='__main__':main()
