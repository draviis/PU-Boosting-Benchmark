"""Fail-closed collection into new tables; originals are never overwritten."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,balanced_accuracy_score,matthews_corrcoef
from worker import digest

HERE=Path(__file__).resolve().parent


def validate_cell(cell,plan):
    folder=HERE/'results'/cell['key']
    marker=json.loads((folder/'complete.json').read_text())
    assert marker['verified'] and marker['key']==cell['key']
    for filename,field in [('results.csv','results_sha256'),('predictions.csv','predictions_sha256'),('provenance.json','provenance_sha256')]:
        assert digest(folder/filename)==marker[field]
    provenance=json.loads((folder/'provenance.json').read_text())
    assert provenance['plan_sha256']==digest(HERE/'plan.json')
    assert provenance['workflow_sha256']==digest(HERE/'worker.py')
    assert provenance['environment_matches_original'] and not provenance['validation_only']
    assert provenance['packages']==cell['packages']
    assert provenance['input_hashes']==cell['files']
    assert provenance['source_sha256']==cell['original_source_sha256']
    assert provenance['repaired_source_sha256']==plan['repaired_sources']
    assert len(provenance['full_grid_fits'])==6
    frame=pd.read_csv(folder/'predictions.csv',float_precision='round_trip')
    old=pd.read_csv(HERE/'inputs'/cell['key']/f'predictions/{cell["method_id"]}.csv',float_precision='round_trip')
    np.testing.assert_array_equal(frame.source_row,old.source_row)
    np.testing.assert_array_equal(frame.y_true,old.y_true)
    result=pd.read_csv(folder/'results.csv',float_precision='round_trip')
    assert len(result)==1
    row=result.iloc[0]
    assert row.method_id==cell['method_id'] and row.seed==cell['seed'] and row.sampling_scheme==cell['scheme']
    assert row.dataset==cell['dataset'] and row.data_fingerprint==cell['data_fingerprint'] and row.status=='ok'
    assert row.c==(.2 if cell['budget']=='c020' else .05)
    assert row.repair_revision=='adapu_rng_v1'
    np.testing.assert_array_equal(frame.prediction,(frame.score>=row.threshold).astype(int))
    y,s,h=frame.y_true,frame.score,frame.prediction
    metrics=dict(AUROC=roc_auc_score(y,s),AUPRC=average_precision_score(y,s),F1=f1_score(y,h,zero_division=0),BalancedAccuracy=balanced_accuracy_score(y,h),MCC=matthews_corrcoef(y,h))
    for key,value in metrics.items():
        np.testing.assert_allclose(row[key],value,rtol=0,atol=1e-12)
    return result,provenance['extension_sha256']


def main():
    plan=json.loads((HERE/'plan.json').read_text())
    original=HERE/'original_combined_results.csv'
    assert digest(original)==(HERE/'original_combined_results.sha256').read_text().strip()
    parts,binaries=[],set()
    for cell in plan['cells']:
        result,binary=validate_cell(cell,plan)
        parts.append(result); binaries.add(binary)
    assert len(binaries)==1, 'Mixed extension builds'
    replacement=pd.concat(parts,ignore_index=True)
    assert len(replacement)==640
    old=pd.read_csv(original,float_precision='round_trip')
    keys=['dataset','c','sampling_scheme','seed','method_id']
    old=old.set_index(keys,verify_integrity=True)
    replacement=replacement.set_index(keys,verify_integrity=True)
    assert replacement.index.isin(old.index).all()
    expected=old.loc[old.index.get_level_values('method_id').isin(['M026','TM026'])]
    assert set(expected.index)==set(replacement.index)
    new=old.copy()
    for column in replacement:
        if column not in new:
            new[column]=np.nan
        # Assignment respects matching MultiIndex keys.
        if replacement[column].dtype=='object':
            new[column]=new[column].astype(object)
        new.loc[replacement.index,column]=replacement[column]
    unchanged=old.index.difference(replacement.index)
    pd.testing.assert_frame_equal(old.loc[unchanged],new.loc[unchanged,old.columns],check_dtype=False,check_exact=True)
    out=HERE/'collected'
    out.mkdir(exist_ok=False)
    new.reset_index().to_csv(out/'combined_results_adapu_rng_v1.csv',index=False)
    effects=expected[['AUROC','AUPRC','F1','BalancedAccuracy','MCC','fit_and_tune_seconds']].join(
        replacement[['AUROC','AUPRC','F1','BalancedAccuracy','MCC','fit_and_tune_seconds']],lsuffix='_original',rsuffix='_repaired')
    for metric in ['AUROC','AUPRC','F1','BalancedAccuracy','MCC']:
        effects[metric+'_change']=effects[metric+'_repaired']-effects[metric+'_original']
    effects.reset_index().to_csv(out/'adapu_before_after.csv',index=False)
    data=new.reset_index()
    data['number']=data.method_id.str[-3:].astype(int)
    summary=data.groupby(['dataset','c','sampling_scheme','number','method'])[['AUROC','AUPRC','F1','BalancedAccuracy','MCC','Brier','LogLoss','fit_and_tune_seconds']].agg(['mean','std','count'])
    summary.to_csv(out/'all_method_summaries.csv')
    ranks=[]
    for metric in ['AUROC','AUPRC']:
        values=data.loc[(data.number>=3)&~data.number.between(11,15)].groupby(['dataset','c','sampling_scheme','number'])[metric].mean().reset_index()
        values['rank']=values.groupby(['dataset','c','sampling_scheme'])[metric].rank(ascending=False,method='average')
        values['metric']=metric
        ranks.append(values)
    ranking=pd.concat(ranks)
    ranking.to_csv(out/'condition_ranks.csv',index=False)
    ranking.groupby(['metric','dataset','number'])['rank'].mean().groupby(['metric','number']).agg(['mean','std','count']).to_csv(out/'overall_average_ranks.csv')
    (out/'complete.json').write_text(json.dumps(dict(verified=True,replaced_rows=640,unchanged_rows=len(unchanged),
                                   total_rows=len(new),original_sha256=digest(original),extension_sha256=list(binaries)[0],
                                   output_sha256=digest(out/'combined_results_adapu_rng_v1.csv')),indent=2))
    print('Verified 640 repaired AdaPU rows; preserved all 16,640 other method rows')


if __name__=='__main__':
    main()
