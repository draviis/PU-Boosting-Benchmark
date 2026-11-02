"""Rebuild processed analyses and figures. This entry point never fits models."""
from pathlib import Path
import argparse,gzip,hashlib,json,os,shutil,subprocess,sys

HERE=Path(__file__).resolve().parent

def verify():
    manifest=json.loads((HERE/'MANIFEST.json').read_text(encoding='utf8'))
    for row in manifest:
        p=HERE/row['path']
        data=p.read_bytes()
        assert hashlib.sha256(data).hexdigest()==row['sha256'],p
        raw=gzip.decompress(data) if row['compression']=='gzip' else data
        assert hashlib.sha256(raw).hexdigest()==row['source_sha256'],p
    print(f'Verified {len(manifest)} archived inputs.',flush=True)
    return manifest

def materialize(out,manifest):
    if out.exists():raise FileExistsError(f'Choose a NEW output directory: {out}')
    if HERE==out or HERE in out.parents:
        raise ValueError('Use an output directory outside the release repository.')
    out.mkdir(parents=True)
    work=out/'workspace';work.mkdir()
    for row in manifest:
        target=work/row['workspace_path']
        assert work.resolve() in target.resolve().parents
        target.parent.mkdir(parents=True,exist_ok=True)
        data=(HERE/row['path']).read_bytes()
        target.write_bytes(gzip.decompress(data) if row['compression']=='gzip' else data)
    return work

def analyse(work,out):
    import numpy as np
    import pandas as pd
    dest=out/'analysis';dest.mkdir()
    read=lambda p:pd.read_csv(p,float_precision='round_trip',low_memory=False)
    t2=work/'mlj_revision/02_fixed_method_and_family_analysis'
    d=read(t2/'repetition_data.csv')
    d=d[d.policy.eq('ranking_archive') & d.metric.isin(['AUROC','AP'])].copy()
    assert len(d)==432000 and d.value.notna().all()
    assert not d.duplicated(['design','scenario_id','rep','method_number','metric']).any()
    reg=pd.DataFrame(json.loads((t2/'scenario_registry.json').read_text()))
    means=d.groupby(['design','scenario_id','method_number','metric']).value.agg(['mean','std','count']).reset_index()
    assert means['count'].eq(100).all()
    means=means.merge(reg[['design','scenario_id','family','family_id']],on=['design','scenario_id'],validate='many_to_one')
    means['MCSE']=means['std']/10
    means['supervised_reference']=means.method_number.le(2)
    means.to_csv(dest/'core_ranking_scenario_means.csv',index=False)
    family=means.groupby(['design','method_number','metric','family','family_id'],as_index=False)['mean'].mean()
    equal=family.groupby(['design','method_number','metric'],as_index=False)['mean'].mean()
    equal['summary_type']='equal_10_family_mean'
    setting=means.groupby(['design','method_number','metric'],as_index=False)['mean'].mean()
    setting['summary_type']='original_40_setting_mean'
    totals=pd.concat([equal,setting],ignore_index=True)
    archived=read(t2/'family_balanced_results.csv')
    check=totals.merge(archived[archived.policy.eq('ranking_archive') & archived.summary_type.isin(totals.summary_type.unique())],on=['design','method_number','metric','summary_type'],validate='one_to_one')
    assert len(check)==216
    np.testing.assert_allclose(check['mean'],check.mean_value,rtol=0,atol=2e-14)
    totals.to_csv(dest/'core_ranking_weighting_comparison.csv',index=False)
    family.to_csv(dest/'core_ranking_within_family.csv',index=False)
    # Fixed same-engine comparisons; all repetition pairing keys remain explicit.
    contrasts=[]
    for key,g in d.groupby(['design','scenario_id','metric']):
        p=g.pivot(index=['rep','seed','data_fingerprint'],columns='method_number',values='value')
        assert len(p)==100
        for n in range(6,11):
            for offset,strategy in [(10,'RN'),(15,'CS')]:
                diff=p[n+offset]-p[n];se=diff.std(ddof=1)/np.sqrt(len(diff));m=diff.mean()
                contrasts.append(dict(design=key[0],scenario_id=key[1],metric=key[2],method_A=n+offset,method_B=n,contrast=strategy+'_minus_naive',n_pairs=len(diff),mean_difference=m,MCSE=se,MC_lower=m-1.96*se,MC_upper=m+1.96*se))
    pd.DataFrame(contrasts).to_csv(dest/'core_same_engine_paired_contrasts.csv',index=False)
    # Execute the archived no-fitting nonlinear aggregations, including all
    # score-hash alias checks and the documented original AdaBoost exceptions.
    t7=work/'mlj_revision/07_nonlinear_simulations'
    t7b=work/'mlj_revision/07b_nonlinear_final_analysis'
    subprocess.run([sys.executable,str(t7/'analyse_results.py')],cwd=work,check=True)
    # Portability-only substitution: parse a safe CSV instead of the local pickle.
    script=(t7b/'analyse.py').read_text(encoding='utf8')
    old="pd.read_pickle(T2/'repetition_data.pkl')"
    assert script.count(old)==1
    script=script.replace(old,"pd.read_csv(T2/'repetition_data.csv',float_precision='round_trip',low_memory=False)")
    ns={'__file__':str(t7b/'analyse.py'),'__name__':'release_analysis'}
    exec(compile(script,str(t7b/'analyse.py'),'exec'),ns)
    ns['main']()
    tables=json.loads((t7b/'analysis_tables.json').read_text())
    for filename,rows in tables.items():
        rebuilt=pd.DataFrame(rows);rebuilt.to_csv(dest/filename,index=False)
        archived=read(t7b/filename)
        assert len(archived)==len(rebuilt),(filename,len(archived),len(rebuilt))
        for col in rebuilt.select_dtypes(include='number'):
            if col in archived:
                np.testing.assert_allclose(rebuilt[col],pd.to_numeric(archived[col]),rtol=0,atol=2e-12,equal_nan=True,err_msg=filename+':'+col)
    nl=read(t7/'nonlinear_results.csv')
    assert len(nl)==32400 and nl.AUROC.isna().sum()==6 and nl.AP.isna().sum()==6
    assert nl[['BA','F1','MCC']].notna().all().all()
    assert nl.hard_metric_model.eq('selected_train_only').all()
    assert not nl.threshold_transfer_used.any() and not nl.test_labels_used_for_selection.any()
    # Real-data primary ranking, equal dataset weights; splits are descriptive.
    rd=read(work/'publication_preparation/eta_scoring_20261013/collected/real_raw.csv')
    assert len(rd)==17280 and rd[['AUROC','AUPRC']].notna().all().all()
    real=rd.groupby(['dataset','sampling_scheme','c','method_id'],as_index=False)[['AUROC','AUPRC']].mean()
    assert len(real)==864
    real.to_csv(dest/'real_data_condition_means.csv',index=False)
    real.groupby(['sampling_scheme','c','method_id'],as_index=False)[['AUROC','AUPRC']].mean().to_csv(dest/'real_data_equal_dataset_means.csv',index=False)
    result=dict(passed=True,model_fits=0,core_ranking_metric_rows=len(d),core_family_and_setting_means_verified=216,
                nonlinear_method_rows=len(nl),nonlinear_missing_refits=6,real_data_method_rows=len(rd),
                nonlinear_tables_verified=list(tables),original_two_sample_AdaBoost_alias_exceptions_retained=True)
    (out/'VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def figures(work,out):
    os.environ['MPLBACKEND']='Agg'
    final=work/'mlj_revision/final_supplement_revision'
    for folder in ['audits','revised_figures','source_data']:(final/folder).mkdir(exist_ok=True)
    env=dict(os.environ,MPLCONFIGDIR=str(out/'matplotlib_cache'))
    subprocess.run([sys.executable,str(final/'source_code/build_figures.py')],cwd=work,env=env,check=True)
    sys.path.insert(0,str(work/'mlj_revision/display_package'))
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import display_figures as fg
    old=work/'mlj_revision/display_package'
    with PdfPages(final/'revised_figures/fig_core_fixed_methods.pdf') as pdf:
        d=pd.read_csv(old/'audit/source_checks/fig_core_fixed_methods_numeric.csv')
        for fig,_,_ in fg.figures('fig_core_fixed_methods',d):pdf.savefig(fig);plt.close(fig)
    # Same real-data code and archived input audit; checksums were reverified above.
    sys.path.insert(0,str(work/'mlj_revision/real_data'))
    import real_data_figures as rf
    for folder in ['main/figures','main/figure_tex','main/data','supplement/figures','supplement/figure_tex','supplement/data','audit/logs']:
        (rf.HOME/folder).mkdir(parents=True,exist_ok=True)
    rf.decomposition();rf.fixed_methods();rf.neural_figure()
    dest=out/'figures';dest.mkdir()
    for p in (final/'revised_figures').glob('*.pdf'):shutil.copy2(p,dest/p.name)
    for folder in ['main','supplement']:
        for p in (rf.HOME/folder/'figures').glob('*.pdf'):shutil.copy2(p,dest/p.name)
    print(f'Figure output: {dest}',flush=True)

def tables(work,out):
    if not shutil.which('pdflatex'):raise RuntimeError('Install a LaTeX distribution with pdflatex first.')
    final=work/'mlj_revision/final_supplement_revision'
    (final/'audits').mkdir(exist_ok=True)
    sys.path.insert(0,str(final/'source_code'))
    import build_tables
    dest=out/'tables';dest.mkdir()
    for p in sorted((final/'revised_tables').glob('*.tex')):
        over=build_tables.compile_table(p.stem)
        if over:print('Layout warnings:',p.name,over)
        for ext in ['.tex','.pdf','.csv']:
            q=p.with_suffix(ext)
            if q.exists():shutil.copy2(q,dest/q.name)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--verify-only',action='store_true')
    ap.add_argument('--output',type=Path)
    ap.add_argument('--materialize-only',action='store_true')
    ap.add_argument('--figures',action='store_true')
    ap.add_argument('--tables',action='store_true')
    args=ap.parse_args();manifest=verify()
    if args.verify_only:return
    if not args.output:ap.error('--output is required unless --verify-only is supplied')
    out=args.output.resolve();work=materialize(out,manifest)
    if args.materialize_only:print(work);return
    analyse(work,out)
    if args.figures:figures(work,out)
    if args.tables:tables(work,out)
    print('Reproduction completed without fitting or refitting models.',flush=True)

if __name__=='__main__':main()
