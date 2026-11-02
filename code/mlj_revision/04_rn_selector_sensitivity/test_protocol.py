"""Meaningful local protocol checks; fake scorers avoid fitting benchmark models."""
from pathlib import Path
import importlib.util,json,ast
import numpy as np
H=Path(__file__).resolve().parent
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
w=module('task4_worker',H/'worker.py');r=module('task3_recovery_test',H/'task3_recovery.py')
w.np=np;r.np=np;w.recovery=r
U=np.array([90,20,40,10,70,60],dtype=np.int32)
np.testing.assert_array_equal(w.size_match_indices(U,np.array([8,8,9,8,8,9]),np.array([.2,.1,.9,.1,.2,.4]),4),[10,20,40,60])
np.testing.assert_array_equal(w.size_match_indices(U,np.ones(6),np.zeros(6),3),[10,20,40])
assert len(w.size_match_indices(U,np.ones(6),np.zeros(6),0))==0
seeds=[w.derived_seed(123,'final_train_plus_validation','random',i) for i in range(1,11)]
assert len(set(seeds))==10 and seeds[0]==w.derived_seed(123,'final_train_plus_validation','random',1)
for seed in seeds:
 a=w.random_subset(U,4,seed);np.testing.assert_array_equal(a,w.random_subset(U,4,seed));assert len(a)==len(set(a))==4 and np.isin(a,U).all()
# A constant spy score equals its quantile: strictly-below must select nobody.
seen=[]
class ConstantSpy:
 def __init__(self,**kw):seen.append(kw)
 def fit(self,X,y):assert set(np.unique(y))=={0,1};return self
 def predict_proba(self,X):return np.tile([.6,.4],(len(X),1))
w.CatBoostClassifier=ConstantSpy
X=np.arange(40,dtype=float).reshape(20,2);s=np.tile([1,0],10)
seed=129;rng=np.random.default_rng(seed);P=np.flatnonzero(s==1)
spies=np.stack([rng.choice(P,size=2,replace=False) for _ in range(10)])
bp=H.parents[1]/'publication_preparation/eta_scoring_20261013/frozen/main/base_phase1_benchmark.py'
nl=w.nonlinear_select(X,s,dict(seed=seed,threads=8),dict(spy_fraction=.2,spy_quantile=.05),dict(np=np),bp,'CatBoost',spies)
assert len(nl['selected'])==0 and np.all(nl['counts']==0)
assert [k['random_seed'] for k in seen]==list(range(seed,seed+10))
assert all(k['iterations']==200 and k['learning_rate']==.05 and k['depth']==4 and k['allow_writing_files']==False for k in seen)
# Same fitted scorer supplies validation and test scores; population differs by design.
fits=[]
class FixedModel:
 def fit(self,X,y):fits.append((X.copy(),y.copy()));return self
 def get_params(self):return dict(test_scorer=True)
ns=dict(make_engine=lambda *args,**kw:FixedModel(),model_score=lambda model,X:X[:,0])
w.balanced_accuracy_score=lambda y,p:float(((p[y==1]==1).mean()+(p[y==0]==0).mean())/2)
out=H/'protocol_test_artifacts';out.mkdir(exist_ok=False)
records=[]
for de,expected in [('one_sample',.65),('two_sample',.7)]:
 for perturb in [0,1]:
  dest=out/(de+str(perturb));dest.mkdir()
  ty=np.array([0,1,0,1]);ty=1-ty if perturb else ty
  z=w.fit_condition(dict(seed=7,design=de,pi_tilde=.25),dict(selector='CatBoost',engine='CatBoost'),ns,'selected_train_only',np.array([[.1],[.2],[.3],[.4]]),np.array([1,0,1,0]),np.array([[.2],[.4],[.6],[.8]]),np.array([1,0,1,0]),np.array([[.1],[.9],[.5],[.7]]),np.array([1],dtype=np.int32),'RN_nonlinear_size_matched',0,None,dict(iterations=200),ty,dest)
  np.testing.assert_allclose(z['threshold'],expected,atol=1e-15)
  assert z['threshold_population_n']==(4 if de=='one_sample' else 2)
  assert not z['threshold_uses_test_labels']
  records.append(dict(design=de,test_labels_permuted=bool(perturb),threshold=z['threshold'],population_n=z['threshold_population_n']))
assert len(fits)==4
for x,y in fits:np.testing.assert_array_equal(y,[1,1,0]);np.testing.assert_array_equal(x[:,0],[.1,.3,.2])
tree=ast.parse((H/'worker.py').read_text())
f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='nonlinear_select')
assert not any(n.id in ['true','y_true','ty'] for n in ast.walk(f) if isinstance(n,ast.Name))
v=dict(status='passed',count_score_index_tie_order=True,exact_size_and_empty_size=True,random_seeds_deterministic_and_distinct=True,strict_cutoff_ties_select_none=True,same_spy_indices_and_seed_structure=True,fixed_CatBoost_selector_config=True,original_P_then_selected_U_training_order=True,threshold_test_label_permutation_invariant=True,separate_validation_populations=records,benchmark_model_fits=0,test_scorer_fits=4)
(H/'protocol_validation.json').write_text(json.dumps(v,indent=2))
print(json.dumps(v,indent=2))
