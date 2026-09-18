"""Frozen within-checkpoint paired statistics, with unit-level inputs."""
import hashlib
import numpy as np
from scipy.stats import rankdata
PREFIX='P2_TRANSPORTED_INTERFACE_FRESH_CONFIRMATION_RC_V2'
SHAMS=('T_signflip','T_wrong_object','T_inplane_sham')
def seed(checkpoint,stratum,sham,procedure):
 return int.from_bytes(hashlib.sha256(f'{PREFIX}|{checkpoint}|{stratum}|{sham}|{procedure}'.encode()).digest()[:8],'big')
def signed_rank(values):
 d=np.asarray(values,dtype=np.float64)
 if d.ndim!=1 or not np.isfinite(d).all():raise ValueError('INVALID_UNIT_EFFECTS')
 d=d[d!=0]
 if not len(d):return dict(p=1.,rank_biserial=0.,Wplus=0.,Wminus=0.,nonzero_n=0)
 ranks2=np.rint(2*rankdata(np.abs(d),method='average')).astype(int)
 total=int(ranks2.sum());pos=int(ranks2[d>0].sum());counts=[0]*(total+1);counts[0]=1
 used=0
 for weight in ranks2:
  for j in range(used,-1,-1):counts[j+weight]+=counts[j]
  used+=int(weight)
 extreme=abs(2*pos-total)
 p=sum(count for s,count in enumerate(counts) if abs(2*s-total)>=extreme)/(2**len(d))
 return dict(p=float(p),rank_biserial=float((2*pos-total)/total),Wplus=pos/2,Wminus=(total-pos)/2,nonzero_n=len(d))
def holm_six(pvalues):
 p=np.asarray(pvalues,dtype=float)
 if p.shape!=(6,) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('EXACT_SIX_TEST_FAMILY_REQUIRED')
 order=np.argsort(p,kind='stable');result=np.empty(6)
 result[order]=np.minimum(1,np.maximum.accumulate(p[order]*(6-np.arange(6))))
 return result.tolist()
def bootstrap(values,rngseed):
 values=np.asarray(values,dtype=float);rng=np.random.Generator(np.random.PCG64(rngseed))
 if not len(values) or not np.isfinite(values).all():raise ValueError('INVALID_BOOTSTRAP_UNITS')
 samples=rng.integers(len(values),size=(20000,len(values)))
 return np.quantile(np.median(values[samples],axis=1),[.025,.975],method='linear').tolist()
def readback(values,checkpoint,stratum,sham):
 x=np.asarray(values,dtype=float);result=signed_rank(x)
 result['bootstrap_median_ci95']=bootstrap(x,seed(checkpoint,stratum,sham,'bootstrap'))
 rng=np.random.Generator(np.random.PCG64(seed(checkpoint,stratum,sham,'permutation')))
 flips=rng.integers(0,2,size=(65536,len(x)))*2-1
 result['signflip_permutation_p']=(1+int(np.sum(np.abs(np.mean(flips*x,axis=1))>=abs(float(x.mean())))))/65537
 result['unit_n']=len(x);result['unit_median_effects']=x.tolist()
 return result
