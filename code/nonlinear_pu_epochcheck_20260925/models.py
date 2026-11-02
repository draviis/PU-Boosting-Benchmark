"""Controlled linear/nonlinear PU capacity comparison. No true labels accepted."""
import copy
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def components(pos, unl, pi, c):
    if not len(pos) or not len(unl):
        raise ValueError('Both positive and unlabeled observations are required')
    if not 0<pi<1 or not 0<=c<1:
        raise ValueError('Invalid prior or labeling probability')
    positive=pi*F.softplus(-pos).mean()
    negative=(1-pi*c)*F.softplus(unl).mean()-pi*(1-c)*F.softplus(pos).mean()
    return positive,negative


def objective(pos,unl,pi,c,nonnegative):
    positive,negative=components(pos,unl,pi,c)
    correction=nonnegative and negative.detach().item()<0
    return (-negative if correction else positive+negative),correction


def architecture(p,width,seed):
    torch.manual_seed(seed)
    return nn.Linear(p,1) if width==0 else nn.Sequential(nn.Linear(p,width),nn.ReLU(),nn.Linear(width,1))


def fit(train_x,train_s,val_x,val_s,test_x,*,pi,c,seed,nonnegative,kind,selection,
        max_epochs=1500,check_every=10,patience=10,widths=(32,128),decays=(1e-4,1e-2)):
    start=time.perf_counter()
    for s in (train_s,val_s):
        if set(np.unique(s))!={0,1}:
            raise ValueError('Both P/U groups required; no seed redraw allowed')
    if kind not in ('linear','mlp') or selection not in ('auc','risk'):
        raise ValueError('Unknown method')
    scaler=StandardScaler().fit(train_x)
    x=torch.tensor(scaler.transform(train_x),dtype=torch.float32)
    v=torch.tensor(scaler.transform(val_x),dtype=torch.float32)
    p=torch.tensor(train_s==1); u=~p
    vp=torch.tensor(val_s==1); vu=~vp
    candidates=[]; best=None
    for width in ((0,) if kind=='linear' else widths):
        for decay in decays:
            model=architecture(x.shape[1],width,seed)
            optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=decay)
            value_best=-np.inf; state=None; best_epoch=None; stale=0; corrections=0; history=[]
            try:
                for epoch in range(1,max_epochs+1):
                    model.train(); optimizer.zero_grad()
                    scores=model(x).flatten()
                    loss,active=objective(scores[p],scores[u],pi,c,nonnegative)
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
                    loss.backward()
                    if not all(torch.isfinite(q.grad).all() for q in model.parameters()):
                        raise FloatingPointError('Nonfinite gradient')
                    optimizer.step(); corrections+=int(active)
                    if epoch%check_every and epoch!=max_epochs: continue
                    model.eval()
                    with torch.no_grad():
                        vs=model(v).flatten()
                        rp,rn=components(vs[vp],vs[vu],pi,c)
                        risk=rp+(rn.clamp_min(0) if nonnegative else rn)
                        score=roc_auc_score(val_s,vs.numpy()) if selection=='auc' else -risk.item()
                    if not np.isfinite(score): raise FloatingPointError('Nonfinite validation criterion')
                    history.append(dict(epoch=epoch,validation_utility=float(score)))
                    if score>value_best:
                        value_best=float(score); state=copy.deepcopy(model.state_dict()); best_epoch=epoch; stale=0
                    else: stale+=1
                    if stale>=patience: break
                if state is None: raise RuntimeError('No valid checkpoint')
                record=dict(width=width,decay=decay,best_epoch=best_epoch,epochs_run=epoch,
                            validation_utility=value_best,corrections=corrections,status='ok',
                            stopped_at_cap=epoch==max_epochs,validation_history=history)
                if best is None or value_best>best['validation_utility']:
                    best={**record,'state':state}
            except (FloatingPointError,ValueError,RuntimeError) as error:
                record=dict(width=width,decay=decay,status='failed',error=repr(error))
            candidates.append(record)
    if best is None: raise RuntimeError(f'All candidates failed: {candidates}')
    chosen=architecture(x.shape[1],best['width'],seed)
    chosen.load_state_dict(best['state'])
    with torch.no_grad(): validation_score=chosen(v).flatten().numpy()
    threshold=float(np.quantile(validation_score if c>0 else validation_score[val_s==0],1-pi))
    dev=torch.cat([x,v]); dp=torch.tensor(np.concatenate([train_s,val_s])==1)
    final=architecture(x.shape[1],best['width'],seed)
    optimizer=torch.optim.Adam(final.parameters(),lr=.001,weight_decay=best['decay'])
    for _ in range(best['best_epoch']):
        optimizer.zero_grad(); scores=final(dev).flatten()
        loss,_=objective(scores[dp],scores[~dp],pi,c,nonnegative)
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite refit objective')
        loss.backward(); optimizer.step()
    final.eval()
    with torch.no_grad():
        score=final(torch.tensor(scaler.transform(test_x),dtype=torch.float32)).flatten().numpy().astype(float)
        probability=torch.sigmoid(torch.tensor(score,dtype=torch.float64)).numpy()
    if not np.isfinite(score).all(): raise FloatingPointError('Nonfinite test score')
    return dict(score=score,probability=probability,threshold=threshold,
                best_params={k:best[k] for k in ('width','decay','best_epoch')},candidates=candidates,
                seconds=time.perf_counter()-start,scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist())
