"""Explicit P->O transport-core successor; no semantic specificity statistics."""
import numpy as np
from src.config import C

def nested(x,mask=None):
 x=np.asarray(x,dtype=float)
 if mask is None:return float(np.median(np.median(x,axis=1)))
 vals=[np.median(row[m]) for row,m in zip(x,mask) if np.any(m)]
 return float(np.median(vals)) if vals else None

def stratum_gates(s):
 g=C['gates']
 return dict(contrastable_coverage=s['contrast_fraction']>=g['contrastable_unit_time_fraction_min_inclusive'],positive_unit_coverage=s['positive_unit_fraction']>=g['positive_unit_median_A_fraction_min_inclusive'],capture=s['capture']>g['transport_capture_min_strict'],recovery=s['recovery'] is not None and s['recovery']>g['median_recovery_min_strict'],raw_T_vs_F=s['eT_contrast'] is not None and s['eF_contrast'] is not None and s['eT_contrast']<s['eF_contrast'],numerical=s['numerical_valid'])

def checkpoint_summary(rows,checkpoint,registry):
 if len(registry)!=C['unit_count'] or len({u['unit_id'] for u in registry})!=len(registry):raise ValueError('REGISTRY_CARDINALITY_FAILURE')
 expected={(u['unit_id'],k) for u in registry for k in C['release_times']};actual=[(r['unit_id'],r['k']) for r in rows]
 if len(actual)!=len(expected) or set(actual)!=expected:raise ValueError('DUPLICATE_OR_MISSING_UNIT_TIME')
 if any(r['checkpoint']!=checkpoint for r in rows):raise ValueError('CHECKPOINT_IDENTITY_DRIFT')
 metadata={u['unit_id']:u for u in registry}
 if any(r['stratum']!=metadata[r['unit_id']]['stratum'] for r in rows):raise ValueError('STRATUM_IDENTITY_DRIFT')
 result=dict(checkpoint=checkpoint,strata={})
 for st in ['S1','S2']:
  units=[u['unit_id'] for u in registry if u['stratum']==st]
  if len(units)!=C['strata_counts'][st]:raise ValueError('STRATUM_CARDINALITY_FAILURE')
  index={(r['unit_id'],r['k']):r for r in rows};grid=[[index[(u,k)] for k in C['release_times']] for u in units]
  errors={kind:np.array([[r['errors'][kind] for r in group] for group in grid]) for kind in C['routes']};cap=np.array([[r['capture'] for r in group] for group in grid])
  numerical=all(r['state_valid'] and r['autonomy_valid'] for group in grid for r in group) and all(np.isfinite(x).all() for x in [*errors.values(),cap])
  if not numerical:raise ValueError('NONFINITE_OR_INVALID_PANEL_NO_EXCLUSION')
  A=errors['F']-errors['O_PATCH'];mask=A>C['contrast_floor'];recovery=np.divide(errors['F']-errors['T'],A,out=np.zeros_like(A),where=mask)
  s=dict(unit_count=len(units),unit_times=len(units)*len(C['release_times']),contrast_fraction=float(mask.mean()),positive_unit_fraction=float(np.mean(np.median(A,axis=1)>0)),capture=nested(cap),recovery=nested(recovery,mask),eT_contrast=nested(errors['T'],mask),eF_contrast=nested(errors['F'],mask),raw_medians={k:nested(v) for k,v in errors.items()},numerical_valid=bool(numerical),contrast_units=int(mask.any(axis=1).sum()),contrast_unit_times=int(mask.sum()),bands={})
  for band,ks in C['time_bands'].items():
   ii=[C['release_times'].index(k) for k in ks];s['bands'][band]=dict(capture=nested(cap[:,ii]),recovery=nested(recovery[:,ii],mask[:,ii]),raw_medians={k:nested(v[:,ii]) for k,v in errors.items()})
  s['gates']=stratum_gates(s);s['supports']=all(s['gates'].values());result['strata'][st]=s
 result['supports']=all(s['supports'] for s in result['strata'].values());return result

def panel_summary(checkpoints,expected_seeds):
 if len(checkpoints)!=3 or [r['checkpoint'] for r in checkpoints]!=expected_seeds or len(set(expected_seeds))!=3:raise ValueError('EXACT_ORDERED_THREE_SEEDS_REQUIRED')
 n=sum(r['supports'] for r in checkpoints)
 return dict(supporting_checkpoints=n,total_checkpoints=3,interpretation=C['panel_labels'][str(n)],seed_replacement=False)
