"""Six specified nonlinear populations, exact likelihood ratios, and SCAR samplers."""
import numpy as np
from scipy.special import i0e,logsumexp,ndtr
SCENARIOS=[
 dict(id='NL01',name='additive_sinusoidal',family='sinusoidal',p=10,q=3,target=.78,parameter='theta',bracket=[0.,3.]),
 dict(id='NL02',name='radial_scale',family='radial',p=10,q=2,target=.80,parameter='r',bracket=[1.,4.]),
 dict(id='NL03',name='pure_XOR',family='XOR',p=10,q=2,target=.84,parameter='a',bracket=[.67,2.]),
 dict(id='NL04',name='threshold_interaction',family='threshold_interaction',p=10,q=3,target=.76,parameter='theta',bracket=[0.,3.]),
 dict(id='NL05',name='sinusoidal_500',family='sinusoidal',p=500,q=3,target=.78,parameter='theta',bracket=[0.,3.]),
 dict(id='NL06',name='heterogeneous_nonlinear',family='heterogeneous_nonlinear',p=10,q=4,target=.86,parameter='theta',bracket=[0.,3.])]
W=np.array([1.,.7,.5]); K=np.array([1,2,1])
SIGMAS=np.array([[[1.,.3],[.3,1.]],[[1.3,-.25],[-.25,.7]]])
INV=np.linalg.inv(SIGMAS); LOGDET=np.linalg.slogdet(SIGMAS)[1]
CHOL=np.linalg.cholesky(SIGMAS)
def config(sn): return SCENARIOS[int(sn)-1]
def streams(sn,phase,*extra):
    seed=np.random.SeedSequence([20261017,7,int(sn),int(phase),*map(int,extra)])
    return [np.random.Generator(np.random.PCG64(s)) for s in seed.spawn(4)]
def h_piece(z):return np.where(z[:,0]>=0,1.,-1.)*np.where(z[:,1]>=.25,1.,-1.)
def h_smooth(z):
    return np.column_stack([np.sin(np.pi*z[:,0])+.5*(z[:,1]**2-1/3),
                            np.cos(np.pi*z[:,1])+.5*z[:,0]*z[:,1]])
def conditional(sn,parameter,y,signal_rng,noise_rng,full=True):
    cfg=config(sn);y=np.asarray(y,dtype=np.int8)
    if not np.isin(y,[0,1]).all() or parameter<0:raise ValueError('invalid class/parameter')
    n=len(y);sign=2*y-1;d=cfg['p'] if full else cfg['q']
    X=np.empty((n,d),dtype=np.float64)
    if sn in [1,5]:
        V=signal_rng.vonmises(sign[:,None]*np.pi/2,parameter*W,size=(n,3))
        cycles=signal_rng.integers(0,K,size=(n,3))
        X[:,:3]=((V+2*np.pi*cycles)/K+np.pi)%(2*np.pi)-np.pi
    elif sn==2:
        if parameter<1:raise ValueError('radial r must be >=1')
        X[:,:2]=signal_rng.standard_normal((n,2))*np.where(y[:,None]==1,parameter,1.)
    elif sn==3:
        C=2*signal_rng.integers(0,2,size=n)-1
        X[:,:2]=parameter*np.column_stack([C,sign*C])+signal_rng.standard_normal((n,2))
    elif sn==4:
        z=signal_rng.uniform(-1,1,(n,2));X[:,:2]=z
        X[:,2]=sign*parameter*h_piece(z)+signal_rng.standard_normal(n)
    elif sn==6:
        z=signal_rng.uniform(-1,1,(n,2));X[:,:2]=z
        eps=signal_rng.standard_normal((n,2))
        X[:,2:4]=sign[:,None]*parameter*h_smooth(z)+np.einsum('nij,nj->ni',CHOL[y],eps)
    else:raise ValueError('invalid scenario')
    if full:X[:,cfg['q']:]=noise_rng.standard_normal((n,cfg['p']-cfg['q']))
    return X
def population(sn,parameter,n,rs,pi=.2,full=True):
    y=rs[2].binomial(1,pi,size=n).astype(np.int8)
    return conditional(sn,parameter,y,rs[0],rs[1],full=full),y
def exact_log_lr(sn,parameter,X):
    if sn in [1,5]:return 2*parameter*(np.sin(X[:,:3]*K)*W).sum(1)
    if sn==2:return -2*np.log(parameter)+.5*(1-parameter**-2)*(X[:,:2]**2).sum(1)
    if sn==3:
        a=parameter*(X[:,0]+X[:,1]);b=parameter*(X[:,0]-X[:,1])
        return np.logaddexp(a,-a)-np.logaddexp(b,-b)
    if sn==4:return 2*parameter*h_piece(X[:,:2])*X[:,2]
    if sn==6:
        mean=parameter*h_smooth(X[:,:2]);t=X[:,2:4]
        dp=t-mean;dn=t+mean
        return -.5*(LOGDET[1]-LOGDET[0])-.5*np.einsum('ni,ij,nj->n',dp,INV[1],dp)+.5*np.einsum('ni,ij,nj->n',dn,INV[0],dn)
    raise ValueError('invalid scenario')
def log_density(sn,parameter,X,y,with_noise=True):
    """Independent density expression for testing score subtraction."""
    from scipy.stats import norm,multivariate_normal,vonmises,uniform
    sign=2*y-1;cfg=config(sn)
    if sn in [1,5]:v=vonmises.logpdf(X[:,:3]*K,parameter*W,loc=sign*np.pi/2).sum(1)
    elif sn==2:v=norm.logpdf(X[:,:2],scale=parameter if y else 1.).sum(1)
    elif sn==3:
        m=np.array([parameter,sign*parameter])
        v=logsumexp(np.column_stack([multivariate_normal.logpdf(X[:,:2],mean=m,cov=np.eye(2)),
                                    multivariate_normal.logpdf(X[:,:2],mean=-m,cov=np.eye(2))]),axis=1)-np.log(2)
    elif sn==4:
        v=uniform.logpdf(X[:,:2],loc=-1,scale=2).sum(1)+norm.logpdf(X[:,2],loc=sign*parameter*h_piece(X[:,:2]))
    elif sn==6:
        v=uniform.logpdf(X[:,:2],loc=-1,scale=2).sum(1)+multivariate_normal.logpdf(
           X[:,2:4]-sign*parameter*h_smooth(X[:,:2]),cov=SIGMAS[y])
    else:raise ValueError('invalid scenario')
    if with_noise and X.shape[1]>cfg['q']:v=v+norm.logpdf(X[:,cfg['q']:]).sum(1)
    return v
def analytic_ceiling(sn,parameter):
    if sn==2:
        if parameter==1:return .5,.5
        r=parameter;q=4*np.log(r)/(1-r**-2)
        return r*r/(1+r*r),.5*(np.exp(-q/(2*r*r))+1-np.exp(-q/2))
    if sn==4:return float(ndtr(np.sqrt(2)*parameter)),float(ndtr(parameter))
    return None,None
def pu_split(sn,parameter,design,rep,split):
    """Model data X/S and evaluation Y separated. Phase 44 reserved for future main data."""
    splitid={'train':0,'validation':1,'test':2}[split];designid={'one_sample':1,'two_sample':2}[design]
    rs=streams(sn,44,designid,rep,splitid)
    if split=='test':
        X,y=population(sn,parameter,10000,rs);S=np.zeros(len(y),dtype=np.int8)
    elif design=='one_sample':
        X,y=population(sn,parameter,5000,rs);S=y*rs[3].binomial(1,.3,len(y)).astype(np.int8)
    else:
        # P and population U draws consume disjoint RNG substreams.
        rp=streams(sn,44,designid,rep,splitid,1);ru=streams(sn,44,designid,rep,splitid,2)
        yp=np.ones(300,dtype=np.int8)
        Xp=conditional(sn,parameter,yp,rp[0],rp[1])
        Xu,yu=population(sn,parameter,4700,ru)
        X=np.vstack([Xp,Xu]);y=np.r_[yp,yu];S=np.r_[np.ones(300,dtype=np.int8),np.zeros(4700,dtype=np.int8)]
    return dict(X=X,S=S),y
def auc_mc(y,score):
    y=np.asarray(y);p=np.sort(score[y==1]);n=np.sort(score[y==0]);np_,nn=len(p),len(n)
    if min(np_,nn)<2:raise ValueError('at least two observations per class')
    pl=np.searchsorted(n,p,'left');pr=np.searchsorted(n,p,'right');vp=(pl+pr)/(2*nn)
    nl=np.searchsorted(p,n,'left');nr=np.searchsorted(p,n,'right');vn=1-(nl+nr)/(2*np_)
    auc=float(vp.mean());assert abs(auc-vn.mean())<1e-12
    se=float(np.sqrt(vp.var(ddof=1)/np_+vn.var(ddof=1)/nn))
    tp=p>=0;tn=n<0;ba=float(.5*(tp.mean()+tn.mean()))
    bse=float(.5*np.sqrt(tp.var(ddof=1)/np_+tn.var(ddof=1)/nn))
    return dict(bayes_auroc=auc,auroc_mcse=se,auroc_mc95_lower=auc-1.96*se,auroc_mc95_upper=auc+1.96*se,
      bayes_ba=ba,ba_mcse=bse,ba_mc95_lower=ba-1.96*bse,ba_mc95_upper=ba+1.96*bse,
      n_positive=np_,n_negative=nn,population_positive_fraction=np_/(np_+nn),
      cross_class_tie_fraction=float((pr-pl).sum()/(np_*nn)),bayes_ba_TPR=float(tp.mean()),bayes_ba_TNR=float(tn.mean()))
