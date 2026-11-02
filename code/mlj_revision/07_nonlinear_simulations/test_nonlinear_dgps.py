"""No learner fits: density/score identities, exact sampling, MC formulas and PU protocol."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import io,json,unittest
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import i0,ndtr
from sklearn.metrics import roc_auc_score
from nonlinear_dgps import *
H=Path(__file__).resolve().parent
DETAILS={}
PARAMS={1:.7,2:2.,3:1.1,4:.5,5:.7,6:.7}
class Tests(unittest.TestCase):
 def test_01_exact_density_ratio(self):
    errs=[]
    for sn,p in PARAMS.items():
        X,y=population(sn,p,2000,streams(sn,33,1))
        a=exact_log_lr(sn,p,X);b=log_density(sn,p,X,1)-log_density(sn,p,X,0)
        error=float(abs(a-b).max());self.assertLess(error,2e-10)
        errs.append(dict(scenario=sn,max_error=error))
    DETAILS['density_ratio_errors']=errs
 def test_02_sinusoidal_normalization_and_sampler_moments(self):
    theta=.7;errors=[]
    for w,k in zip(W,K):
        integral=quad(lambda x:np.exp(theta*w*np.sin(k*x))/(2*np.pi*i0(theta*w)),-np.pi,np.pi,epsabs=1e-12)[0]
        self.assertAlmostEqual(integral,1.,places=11)
    from scipy.special import i1
    for y in [0,1]:
        rs=streams(1,33,2,y);X=conditional(1,theta,np.full(100000,y,dtype=np.int8),rs[0],rs[1],full=False)
        obs=np.sin(X*K).mean(0);expected=(2*y-1)*i1(theta*W)/i0(theta*W)
        delta=obs-expected;self.assertLess(abs(delta).max(),.012)
        self.assertTrue(((X>=-np.pi)&(X<np.pi)).all())
        errors.append(dict(y=y,observed=obs.tolist(),expected=expected.tolist()))
    DETAILS['sinusoidal_moments']=errors
 def test_03_radial_and_piecewise_analytic(self):
    vals=[]
    for sn in [2,4]:
        theta=PARAMS[sn];X,y=population(sn,theta,200000,streams(sn,33,3),full=False)
        m=auc_mc(y,exact_log_lr(sn,theta,X));a,b=analytic_ceiling(sn,theta)
        self.assertLess(abs(m['bayes_auroc']-a),5*m['auroc_mcse'])
        self.assertLess(abs(m['bayes_ba']-b),5*m['ba_mcse'])
        vals.append(dict(scenario=sn,analytic_auroc=a,analytic_ba=b,**m))
    DETAILS['analytic_mc_checks']=vals
 def test_04_high_dimensional_noise_cancels(self):
    rs=streams(1,33,4);X,y=population(1,.7,1000,rs)
    rs=streams(1,33,4);Z,yy=population(5,.7,1000,rs)
    np.testing.assert_array_equal(y,yy);np.testing.assert_array_equal(X[:,:3],Z[:,:3])
    np.testing.assert_array_equal(exact_log_lr(1,.7,X),exact_log_lr(5,.7,Z))
 def test_05_monte_carlo_auc_ties_and_orientation(self):
    y=np.array([1,0,1,0,1,0,1,0]);score=np.array([0,0,2,1,1,2,3,1.])
    r=auc_mc(y,score);self.assertAlmostEqual(r['bayes_auroc'],roc_auc_score(y,score),places=15)
    self.assertGreater(r['cross_class_tie_fraction'],0)
    self.assertAlmostEqual(auc_mc(y,-score)['bayes_auroc'],1-r['bayes_auroc'])
 def test_06_PU_protocol_and_reproducibility(self):
    info=[]
    for design in ['one_sample','two_sample']:
        arrays=[]
        for split in ['train','validation','test']:
            d,y=pu_split(1,.7,design,9999,split)
            self.assertEqual(set(d),{'X','S'})
            self.assertFalse(np.any((d['S']==1)&(y==0)))
            self.assertEqual(d['X'].shape,(10000 if split=='test' else 5000,10))
            arrays.append(d['X'])
            if split!='test':
                if design=='two_sample':
                    self.assertEqual(int(d['S'].sum()),300);self.assertEqual(int((d['S']==0).sum()),4700)
                    # U must retain hidden positives.
                    self.assertGreater(y[d['S']==0].sum(),800)
                else:self.assertTrue(200<int(d['S'].sum())<400)
            dd,yy=pu_split(1,.7,design,9999,split)
            np.testing.assert_array_equal(dd['X'],d['X']);np.testing.assert_array_equal(yy,y)
            info.append(dict(design=design,split=split,n=len(y),observed_P=int(d['S'].sum())))
        self.assertFalse(np.array_equal(arrays[0],arrays[1]))
    DETAILS['PU_sampler_checks']=info
 def test_07_covariance_and_noise_dimensions(self):
    self.assertTrue((np.linalg.eigvalsh(SIGMAS)>0).all())
    for sn in range(1,7):
        X,y=population(sn,PARAMS[sn],400,streams(sn,33,7))
        self.assertEqual(X.shape,(400,config(sn)['p']))
        self.assertTrue(np.isfinite(exact_log_lr(sn,PARAMS[sn],X)).all())
    # Nonlinear conditional mean integrates to zero by symmetry / centering.
    z=streams(6,33,8)[0].uniform(-1,1,(200000,2))
    self.assertLess(abs(h_smooth(z).mean(0)).max(),.01)
if __name__=='__main__':
 out=io.StringIO();r=unittest.TextTestRunner(stream=out,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
 dest=H/'dgp_unit_validation.json'
 with dest.open('x') as f:json.dump(dict(passed=r.wasSuccessful(),tests=r.testsRun,details=DETAILS,log=out.getvalue()),f,indent=2)
 print(out.getvalue());raise SystemExit(0 if r.wasSuccessful() else 1)
