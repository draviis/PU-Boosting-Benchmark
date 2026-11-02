"""Task 4 extensions: reuse only byte-identical existing inputs/configurations."""
from pathlib import Path
import json,shutil,sys
import worker as w
H=Path(__file__).resolve().parent
original_fit=w.fit_condition;original_select=w.nonlinear_select;original_run=w.run
events=[]
def fit_condition(c,j,ns,stage,X,s,Xval,sval,Xtest,selected,condition,draw,seed_draw,params,ty,out):
 if j['selector']=='XGBoost' and condition in ['RN_logistic','RN_random_size_matched','Oracle_true_negative_size_matched','All_U_RN_fixed_final']:
  source=H/'results'/(c['key']+'__'+j['engine']+'__CatBoost')
  complete=source/'complete.json'
  if complete.exists():
   meta=json.loads(complete.read_text())
   assert meta['data_fingerprint']==c['fingerprint']
   name=stage+'__'+condition+f'__{draw:02d}';jp=source/(name+'.json');ap=source/(name+'.npz')
   if jp.exists():
    d=json.loads(jp.read_text())
    assert d['selected_set_sha256']==w.ah(selected) and d['final_params']==params and d['final_seed']==c['seed'] and d['draw_seed']==seed_draw
    assert w.sha(ap)==d['arrays_sha256']
    dest=out/ap.name;record=out/jp.name
    if not dest.exists():shutil.copyfile(ap,dest)
    else:assert w.sha(dest)==d['arrays_sha256']
    d.update(source='Task4_reused_identical_fit',selector=j['selector'],reuse_from=ap.relative_to(H).as_posix(),reuse_record_sha256=w.sha(jp),reuse_wrapper_sha256=w.sha(Path(__file__)))
    if not record.exists():w.write_json(record,d)
    else:assert json.loads(record.read_text())==w.clean(d)
    events.append(dict(kind='final_fit',source=d['reuse_from'],condition=condition,draw=draw))
    return d
 return original_fit(c,j,ns,stage,X,s,Xval,sval,Xtest,selected,condition,draw,seed_draw,params,ty,out)
def nonlinear_select(X,s,c,params,ns,bp,selector,old_spies):
 if selector=='CatBoost':
  source=H/'results'/(c['key']+'__CatBoost__CatBoost')
  for stage in ['final_train_plus_validation','selected_train_only']:
   jp=source/(stage+'__nonlinear_selection.json');ap=source/(stage+'__nonlinear_selection.npz')
   if jp.exists():
    sm=json.loads(jp.read_text())
    if sm['spy_fraction']!=params['spy_fraction'] or sm['spy_quantile']!=params['spy_quantile']:continue
    if sm['source_X_sha256']!=w.ah(X) or sm['source_S_sha256']!=w.ah(s):continue
    assert sm['seed']==c['seed'] and sm['source_data_fingerprint']==c['fingerprint'] and w.sha(ap)==sm['arrays_sha256']
    with w.np.load(ap) as z:nl={k:z[k].copy() for k in z.files}
    w.np.testing.assert_array_equal(nl['spies'],old_spies)
    nl['configs']=sm['configs'];nl['original_selector_ast_sha256']=sm['original_selector_ast_sha256']
    events.append(dict(kind='selector_ten_fits',source=ap.relative_to(H).as_posix(),source_sha256=sm['arrays_sha256']))
    return nl
 return original_select(X,s,c,params,ns,bp,selector,old_spies)
def run(c,j):
 events.clear()
 original_run(c,j)
 out=H/'results'/j['job_key'];p=out/'extension_execution.json'
 if not p.exists():w.write_json(p,dict(extension_wrapper_sha256=w.sha(Path(__file__)),base_worker_sha256=w.sha(H/'worker.py'),reused=events,new_conditions_unchanged=True))
w.fit_condition=fit_condition;w.nonlinear_select=nonlinear_select;w.run=run
if __name__=='__main__':w.main()
