"""Final nonlinear analysis using only existing Task 2 and Task 7 outputs. No fit imports."""
from pathlib import Path
import json, math, hashlib
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parent;R=H.parents[1]
T2=H.with_name('02_fixed_method_and_family_analysis');T7=H.with_name('07_nonlinear_simulations')
METRICS=['AUROC','AP'];ENG=['AdaBoost','GBM','XGBoost','LightGBM','CatBoost']
IDS=list(range(3,11))+list(range(16,28))
DGP={'NL01':'Sinusoidal','NL02':'Radial','NL03':'Pure XOR','NL04':'Threshold interaction','NL05':'Sinusoidal + 497 noise variables','NL06':'Nonlinear heterogeneous covariance'}
OLD={1:'Gaussian reference',26:'Gaussian, p=500',38:'Curved boundary',39:'Feature interaction + main effect',40:'Unequal covariance'}
def save(n,x):
    with (H/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def name(n):
    if n<6:return {3:'Corrected logistic',4:'uPU logistic',5:'nnPU logistic'}[n]
    if n<26:return ['Naive','Corrected','RN','CS'][(n-6)//5]+' '+ENG[(n-6)%5]
    return 'AdaPU' if n==26 else 'PU gradient boosting'
def records(d):return json.loads(d.to_json(orient='records',double_precision=15))
def stats(v):
    v=np.asarray(v,dtype=float);v=v[np.isfinite(v)]
    se=float(np.std(v,ddof=1)/np.sqrt(len(v)))
    return dict(n=int(len(v)),mean=float(v.mean()),MCSE=se,MC_lower=float(v.mean()-1.96*se),MC_upper=float(v.mean()+1.96*se))
def main():
    raw7=json.loads((T7/'nonlinear_results.json').read_text());new=pd.DataFrame(raw7)
    new['method_number']=new.method_id.str[-3:].astype(int)
    old=pd.read_pickle(T2/'repetition_data.pkl');old=old[old.policy.eq('ranking_archive')&old.metric.isin(METRICS)].copy()
    assert len(new)==32400 and len(old)==432000
    assert not new.duplicated(['case','method_number']).any()
    assert new.groupby('case').model_seed.nunique().eq(1).all()
    assert new.groupby(['scenario_id','design','method_number']).rep.nunique().eq(100).all()
    assert new.AUROC.isna().sum()==new.AP.isna().sum()==6
    reg={(r['design'],r['scenario_id']):r for r in json.loads((T2/'scenario_registry.json').read_text())}
    bayes2={(r['design'],r['scenario_id']):r for r in json.loads((T2/'bayes_results.json').read_text())}
    bayes7={r['scenario_id']:r for r in json.loads((T7/'nonlinear_dgp_calibration.json').read_text())}
    inv={(r['case'],r['method_id']):r for r in json.loads((T7/'score_archive_inventory.json').read_text())}
    validation=json.loads((T7/'main_validation.json').read_text());assert validation['passed'] and len(validation['matched_naive_corrected'])==6000
    assert len(inv)==32400
    aliases=[];exceptions=[]
    for (des,sc),g in new.groupby(['design','scenario_id']):
        for n in range(6,11):
            a=g[g.method_number.eq(n)].set_index('rep');b=g[g.method_number.eq(n+5)].set_index('rep')
            assert set(a.index)==set(b.index)==set(range(100))
            for rep in a.index:
                ia=inv[(a.loc[rep,'case'],f'M{n:03d}')];ib=inv[(b.loc[rep,'case'],f'M{n+5:03d}')]
                assert all(ia['score_hashes'][k]==ib['score_hashes'][k] for k in ['selected_val_raw','selected_test_raw','refit_val_raw','refit_test_raw'])
            for metric in METRICS:
                delta=a[metric]-b[metric];assert (delta==0).all()
                aliases.append(dict(task='Task7',design=des,scenario_id=sc,engine=ENG[n-6],metric=metric,naive_id=f'M{n:03d}',corrected_id=f'M{n+5:03d}',paired_repetitions=100,exact_metric_matches=100,max_abs_metric_difference=0.,within_roundoff=True,raw_score_hash_identity_verified=True,collapsed_in_rank_catalog=True,verification='All four saved score hashes identical; exact repetition metrics'))
    for (des,sc,metric),g in old.groupby(['design','scenario_id','metric']):
        p=g.pivot(index=['rep','seed','data_fingerprint'],columns='method_number',values='value')
        assert len(p)==100 and p.notna().all().all()
        for n in range(6,11):
            delta=p[n]-p[n+5];exact=int((delta==0).sum());mx=float(delta.abs().max())
            collapsed=not(des=='two_sample' and n==6)
            if collapsed:assert mx<=1e-14
            aliases.append(dict(task='Task2',design=des,scenario_id=sc,engine=ENG[n-6],metric=metric,naive_id=('M' if des=='one_sample' else 'TM')+f'{n:03d}',corrected_id=('M' if des=='one_sample' else 'TM')+f'{n+5:03d}',paired_repetitions=100,exact_metric_matches=exact,max_abs_metric_difference=mx,within_roundoff=mx<=1e-14,raw_score_hash_identity_verified=None,collapsed_in_rank_catalog=collapsed,verification='Archived repetition metrics only; tolerance 1e-14; raw scores not present in Task2 outputs'))
            for (rep,seed,fp),diff in delta[delta.abs()>1e-14].items():
                exceptions.append(dict(task='Task2',design=des,scenario_id=sc,repetition=int(rep),engine=ENG[n-6],metric=metric,naive_value=float(p.loc[(rep,seed,fp),n]),corrected_value=float(p.loc[(rep,seed,fp),n+5]),naive_minus_corrected=float(diff),decision='Keep both AdaBoost entries in all 40 original two-sample ranking catalogs'))
    assert len(exceptions)==4 and {(x['scenario_id'],x['repetition']) for x in exceptions}=={('TS001',85),('TS026',82)}
    ranks_new=[];ranks_old=[];summ7={}
    for (des,sc,mid),g in new.groupby(['design','scenario_id','method_number']):
        if mid<3:continue
        summ7[(des,sc,mid)]={metric:stats(g[metric]) for metric in METRICS}
    for task,data,out in [('Task7',new,ranks_new),('Task2',old,ranks_old)]:
        for (des,sc),g in data.groupby(['design','scenario_id']):
            ids=sorted(IDS+([11] if task=='Task2' and des=='two_sample' else []))
            for metric in METRICS:
                group=[]
                for n in ids:
                    ss=stats(g.loc[g.method_number.eq(n),metric] if task=='Task7' else g.loc[g.method_number.eq(n)&g.metric.eq(metric),'value'])
                    collapse=6<=n<=10 and not(task=='Task2' and des=='two_sample' and n==6)
                    label='Naive/Corrected '+ENG[n-6] if collapse else name(n)
                    group.append(dict(task=task,design=des,scenario_id=sc,scenario=DGP[sc] if task=='Task7' else reg[(des,sc)]['scenario_label'],family=g.family.iloc[0] if task=='Task7' else reg[(des,sc)]['family'],retained_method_number=n,retained_method_id=('TM' if task=='Task2' and des=='two_sample' else 'M')+f'{n:03d}',ranking_entry=label,alias_method_numbers=f'{n:03d}/{n+5:03d}' if collapse else f'{n:03d}',metric=metric,mean_value=ss['mean'],n_repetitions=ss['n'],n_missing=100-ss['n'],MCSE=ss['MCSE'],rank_catalog_n=len(ids),alias_values_averaged=False,rank_basis='descending repetition mean; average ties after 12-decimal rounding',conditional_on_refit_success=ss['n']<100))
                rk=pd.Series([r['mean_value'] for r in group]).round(12).rank(ascending=False,method='average').tolist()
                for r,k in zip(group,rk):r['rank']=k
                assert abs(sum(rk)-len(ids)*(len(ids)+1)/2)<1e-12
                out.extend(group)
    decomp=[]
    for (des,sc),g in new.groupby(['design','scenario_id']):
        for engine,n in [('CatBoost',10),('XGBoost',8)]:
            for kind,a,b in [('learner_gain',n,3),('CS_additional_gain',n+15,n),('RN_additional_gain',n+10,n)]:
                aa=g[g.method_number.eq(a)].set_index(['rep','model_seed'])
                bb=g[g.method_number.eq(b)].set_index(['rep','model_seed'])
                assert aa.index.equals(bb.index) or set(aa.index)==set(bb.index)
                for metric in METRICS:
                    diff=(aa[metric]-bb[metric]).sort_index();ss=stats(diff)
                    decomp.append(dict(scenario_id=sc,DGP=DGP[sc],design=des,engine=engine,contrast=kind,method_A=f'M{a:03d}',method_A_name=name(a),method_B=f'M{b:03d}',method_B_name=name(b),metric=metric,n_pairs=ss['n'],n_expected=100,n_missing=100-ss['n'],excluded_repetitions=';'.join(str(int(t[0])) for t in diff[diff.isna()].index),mean_difference=ss['mean'],MCSE=ss['MCSE'],MC_lower=ss['MC_lower'],MC_upper=ss['MC_upper'],mean_difference_pp=100*ss['mean'],MCSE_pp=100*ss['MCSE'],MC_lower_pp=100*ss['MC_lower'],MC_upper_pp=100*ss['MC_upper'],conditional_on_joint_refit_success=ss['n']<100,paired_on='scenario/design/repetition/model_seed'))
    dl={(r['design'],r['scenario_id'],r['engine'],r['contrast'],r['metric']):r for r in decomp}
    fixed={'corrected_logistic':3,'nnPU_logistic':5,'naive_CatBoost':10,'CS_CatBoost':25,'PU_gradient_boosting':27}
    comp=[]
    for des in ['one_sample','two_sample']:
        for task,scs in [('Task2',[('S' if des=='one_sample' else 'TS')+f'{i:03d}' for i in OLD]),('Task7',list(DGP))]:
            for sc in scs:
                bc=bayes2[(des,sc)] if task=='Task2' else bayes7[sc]
                row=dict(design=des,context='original' if task=='Task2' else 'nonlinear',scenario_id=sc,DGP=OLD[int(sc[-3:])] if task=='Task2' else DGP[sc],p=int(reg[(des,sc)]['p']) if task=='Task2' else int(bc['p']),prevalence=.2,Bayes_AUROC=bc['Bayes_AUROC'] if task=='Task2' else bc['bayes_auroc'],Bayes_AUROC_MCSE=bc.get('Bayes_AUROC_MCSE',0.) if task=='Task2' else bc['auroc_mcse'],Bayes_source=bc['AUROC_source'] if task=='Task2' else 'existing exact-LR independent Monte Carlo calibration',comparison_type='different DGP contexts, not paired old-versus-new contrasts')
                for label,n in fixed.items():
                    for metric in METRICS:
                        ss=stats(old.loc[old.design.eq(des)&old.scenario_id.eq(sc)&old.method_number.eq(n)&old.metric.eq(metric),'value']) if task=='Task2' else summ7[(des,sc,n)][metric]
                        row[label+'_'+metric]=ss['mean'];row[label+'_'+metric+'_MCSE']=ss['MCSE'];row[label+'_'+metric+'_n']=ss['n']
                        if metric=='AUROC':row[label+'_AUROC_regret']=row['Bayes_AUROC']-ss['mean']
                comp.append(row)
    headline=[]
    for des in ['one_sample','two_sample']:
        for sc in DGP:
            row=dict(design=des,scenario_id=sc,DGP=DGP[sc],Bayes_AUROC=bayes7[sc]['bayes_auroc'],Bayes_AUROC_MCSE=bayes7[sc]['auroc_mcse'])
            for label,n in [('corrected_logistic',3),('naive_CatBoost',10),('CS_CatBoost',25),('RN_CatBoost',20),('PU_gradient_boosting',27)]:
                ss=summ7[(des,sc,n)]['AUROC']
                row[label+'_AUROC']=ss['mean'];row[label+'_n']=ss['n']
            for kind in ['learner_gain','CS_additional_gain','RN_additional_gain']:
                z=dl[(des,sc,'CatBoost',kind,'AUROC')]
                for col in ['mean_difference_pp','MCSE_pp','MC_lower_pp','MC_upper_pp','n_pairs']:row[kind+'_'+col]=z[col]
            headline.append(row)
    failures=[]
    for _,r in new[new.AUROC.isna()].iterrows():
        assert not bool(r.get('refit_available',True)) and r.refit_failure_reason=='empty_RN_refit'
        failures.append(dict(scenario=r.scenario_id,design=r.design,repetition=int(r.rep),repetition_indexing='Task7 saved zero-based index',method_id=r.method_id,method=name(r.method_number),reason='Selected RN settings yielded zero reliable negatives in final train+validation refit',AUROC_available=False,AP_available=False,selected_BA_F1_MCC_available=True,imputed=False))
    assert len(failures)==6
    aggregate=[]
    for des in ['one_sample','two_sample']:
        for eng in ['CatBoost','XGBoost']:
            for kind in ['learner_gain','CS_additional_gain','RN_additional_gain']:
                for metric in METRICS:
                    for rule,weights in [('six_DGP_mean',[1/6]*6),('equal_five_mechanism_mean',[.1,.2,.2,.2,.1,.2])]:
                        cells=[dl[(des,sc,eng,kind,metric)] for sc in DGP]
                        mean=sum(w*x['mean_difference'] for w,x in zip(weights,cells));se=math.sqrt(sum((w*x['MCSE'])**2 for w,x in zip(weights,cells)))
                        aggregate.append(dict(design=des,engine=eng,contrast=kind,metric=metric,aggregation=rule,mean=mean,MCSE=se,MC_lower=mean-1.96*se,MC_upper=mean+1.96*se,n_pairs=sum(x['n_pairs'] for x in cells)))
    dimensions=[];gaps=[]
    for des in ['one_sample','two_sample']:
        for n in IDS:
            for metric in METRICS:
                a=summ7[(des,'NL05',n)][metric];b=summ7[(des,'NL01',n)][metric]
                difference=a['mean']-b['mean'];se=math.hypot(a['MCSE'],b['MCSE'])
                dimensions.append(dict(design=des,method_number=n,method=name(n),metric=metric,NL01=b['mean'],NL05=a['mean'],difference=difference,MCSE=se,MC_lower=difference-1.96*se,MC_upper=difference+1.96*se,paired=False,reason='independent DGP streams'))
            for sc in DGP:
                z=summ7[(des,sc,n)]['AUROC'];bc=bayes7[sc]
                gaps.append(dict(design=des,scenario_id=sc,method_number=n,method=name(n),AUROC=z['mean'],method_MCSE=z['MCSE'],n=z['n'],Bayes_AUROC=bc['bayes_auroc'],Bayes_MCSE=bc['auroc_mcse'],regret=bc['bayes_auroc']-z['mean'],regret_MCSE_including_independent_calibration=math.hypot(z['MCSE'],bc['auroc_mcse'])))
    tables={'nonlinear_unique_score_model_ranks.csv':ranks_new,'original_unique_score_model_ranks.csv':ranks_old,'nonlinear_learner_strategy_decomposition.csv':decomp,'original_vs_nonlinear_comparison.csv':comp,'nonlinear_headline_results.csv':headline,'unavailable_RN_refits.csv':failures,'ranking_alias_verification.csv':aliases,'ranking_alias_exceptions.csv':exceptions}
    save('analysis_tables.json',tables)
    save('supporting_statistics.json',dict(aggregate_contrasts=aggregate,dimension_comparisons=dimensions,Bayes_gaps=gaps))
    save('analysis_validation.json',dict(passed=True,model_fits=0,model_refits=0,input_tasks=[2,7],Task7_exact_alias_case_pairs=6000,Task2_metric_alias_comparisons=80000,Task2_nonroundoff_exceptions=exceptions,Task2_raw_scores_not_available_in_scope=True,Task7_rank_universe=20,Task2_rank_universe={'one_sample':20,'two_sample':21},n_missing_refits=6,paired_contrast_rows=len(decomp),paired_contrast_rows_with_99_pairs=sum(r['n_pairs']==99 for r in decomp),all_other_contrasts_100_pairs=all(r['n_pairs'] in [99,100] for r in decomp),table_rows={k:len(v) for k,v in tables.items()},no_scores_averaged=True))
    print(json.dumps({k:len(v) for k,v in tables.items()}))
if __name__=='__main__':main()
