"""Check recovered pilot against original fitting on the same feasible grid."""
from pathlib import Path
import os,sys,json,hashlib,inspect
H=Path(__file__).resolve().parent
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','PU_BENCHMARK_N_JOBS']:os.environ[k]='8'
sys.path.insert(0,str(H/'sources/two_sample'))
import numpy as np
import two_sample_benchmark as module
from nonlinear_dgps import pu_split,config
from threadpoolctl import threadpool_limits
base=module.base

def main():
    case=json.loads((H/'main_cases.json').read_text())[701]
    assert case['key']=='NL02_two_sample_r001'
    parameter=json.loads((H/'frozen_dgp_parameters.json').read_text())['parameters']['NL02']['value']
    splits={}
    for short,long in [('train','train'),('val','validation'),('test','test')]:
        v,y=pu_split(2,parameter,'two_sample',1,long)
        splits[short]=base.PUSplit(X=v['X'],s=v['S'],y_true=np.full(len(y),-1,dtype=np.int8))
    cfg=config(2)
    data=base.PUExperiment(**splits,pi=.2,c=0.,p=cfg['p'],delta=np.nan,delta_sq=np.nan,signal=np.nan,n_signal=cfg['q'],rho=np.nan,seed=case['model_seed'])
    module._ACTIVE_VALIDATION_U_MASK=data.val.s==0
    original_grid=base.STRATEGY_GRIDS['RN'];valid=[];counts=[]
    for par in original_grid:
        x,y,w=base.prepare_strategy_training_data(data.train,'RN',par,seed=data.seed)
        counts.append(dict(params=par,n_RN=int((y==0).sum())))
        if len(np.unique(y))==2:valid.append(par)
    assert len(valid)==3
    base.STRATEGY_GRIDS['RN']=valid
    result=[]
    def evaluate(method,y_test,score_test,threshold,probability=None,extra=None):
        assert np.all(y_test==-1)
        loc=inspect.currentframe().f_back.f_locals
        return dict(selected_val_raw=loc['best']['raw_val'].copy(),refit_test_raw=np.asarray(score_test).copy(),refit_val_raw=base.model_score(loc['final'],data.val.X),extra=extra)
    base.evaluate_test=evaluate
    for block,mid,engine in [(1,'M016','AdaBoost'),(2,'M017','GBM'),(3,'M018','XGBoost'),(5,'M020','CatBoost')]:
        folder=H/'results'/case['key']/('block'+str(block))
        actual=json.loads((folder/(mid+'.json')).read_text())
        plain=module.fit_one_boosting_cell(data,'RN',engine)
        assert plain['extra']==actual['selection']
        with np.load(folder/(mid+'.npz')) as z:
            for key in ['selected_val_raw','refit_val_raw','refit_test_raw']:np.testing.assert_allclose(z[key],plain[key],rtol=0,atol=1e-12)
        result.append(dict(method=mid,selected_params_equal=True,selected_validation_scores_equal=True,refitted_validation_test_scores_equal=True))
    out=dict(passed=True,case=case['key'],reference='Unmodified original fitting function with the identical feasible subset of the original tuning grid',training_RN_counts=counts,methods=result)
    with (H/'rn_recovery_scientific_parity.json').open('x') as f:json.dump(out,f,indent=2)
    print(json.dumps(out),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=8):main()
