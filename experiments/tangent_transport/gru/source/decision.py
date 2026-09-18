"""Checkpoint-first gates. Synthetic fixtures are tests, not scientific evidence."""
import numpy as np
from paired_statistics import SHAMS,readback,holm_six
from runtime_config import C
def nested(x,mask=None):
 x=np.asarray(x,dtype=float)
 if mask is None:return float(np.median(np.median(x,axis=1)))
 vals=[np.median(row[m]) for row,m in zip(x,mask) if np.any(m)]
 return float(np.median(vals)) if vals else None
def stratum_gates(s):
 g=C['gates']
 return dict(contrastability=s['contrast_fraction']>=g['contrastable_unit_time_fraction_min_inclusive'] and s['positive_unit_fraction']>=g['positive_unit_median_A_fraction_min_inclusive'],
 capture=s['capture']>g['transport_capture_min_strict'],
 functional=s['recovery'] is not None and s['recovery']>g['median_recovery_min_strict'] and s['eT_contrast']<s['eF_contrast'],
 specificity=all(s['effects'][sh]>g['paired_effect_min_strict'] and s['wins'][sh]>g['correct_win_fraction_min_strict'] for sh in SHAMS),
 statistics=all(s['statistics'][sh]['holm_p']<g['holm_p_max_strict'] for sh in SHAMS),
 numerical=s['numerical_valid'])
def checkpoint_summary(rows,cp):
 result={'checkpoint':cp,'strata':{}}
 for st in ['S1','S2']:
  rs=[r for r in rows if r['stratum']==st];units=list(dict.fromkeys(r['unit_id'] for r in rs))
  grid=[[next(r for r in rs if r['unit_id']==u and r['k']==k) for k in C['release_times']] for u in units]
  errors={kind:np.array([[r['errors'][kind] for r in group] for group in grid]) for kind in C['routes']}
  cap=np.array([[r['capture'] for r in group] for group in grid])
  numerical=all(r['state_valid'] for r in rs) and all(np.isfinite(x).all() for x in list(errors.values())+[cap])
  if not numerical:raise ValueError('NONFINITE_OR_INVALID_PANEL_NO_EXCLUSION')
  A=errors['F']-errors['P'];mask=A>C['contrast_floor']
  rec=np.divide(errors['F']-errors['T_correct'],A,out=np.zeros_like(A),where=mask)
  effects={sh:errors[sh]-errors['T_correct'] for sh in SHAMS}
  s=dict(unit_count=len(units),unit_times=len(rs),contrast_fraction=float(mask.mean()),positive_unit_fraction=float(np.mean(np.median(A,axis=1)>0)),
   capture=nested(cap),recovery=nested(rec,mask),eT_contrast=nested(errors['T_correct'],mask),eF_contrast=nested(errors['F'],mask),
   raw_medians={k:nested(v) for k,v in errors.items()},effects={k:nested(v) for k,v in effects.items()},
   wins={k:float(np.mean(v>0)) for k,v in effects.items()},numerical_valid=bool(numerical),
   contrast_units=int(np.sum(mask.any(axis=1))),contrast_unit_times=int(mask.sum()),
   statistics={sh:readback(np.median(effects[sh],axis=1),cp,st,sh) for sh in SHAMS},bands={})
  for band,ks in C['time_bands'].items():
   inds=[C['release_times'].index(k) for k in ks]
   s['bands'][band]=dict(capture=nested(cap[:,inds]),recovery=nested(rec[:,inds],mask[:,inds]),effects={sh:nested(v[:,inds]) for sh,v in effects.items()},
      raw_medians={kind:nested(v[:,inds]) for kind,v in errors.items()})
  result['strata'][st]=s
 family=[result['strata'][st]['statistics'][sh] for st in ['S1','S2'] for sh in SHAMS]
 for test,p in zip(family,holm_six([test['p'] for test in family])):test['holm_p']=p
 for s in result['strata'].values():s['gates']=stratum_gates(s);s['supports']=all(s['gates'].values())
 result['supports']=all(s['supports'] for s in result['strata'].values())
 return result
def panel_summary(checkpoints):
 if len(checkpoints)!=3 or len({x['checkpoint'] for x in checkpoints})!=3:raise ValueError('EXACT_THREE_CHECKPOINTS_REQUIRED')
 g=C['gates'];numerical=all(x['strata'][st]['numerical_valid'] for x in checkpoints for st in ['S1','S2'])
 support=sum(bool(x['supports']) for x in checkpoints)
 medians={st:dict(capture=float(np.median([x['strata'][st]['capture'] for x in checkpoints])),
  effects={sh:float(np.median([x['strata'][st]['effects'][sh] for x in checkpoints])) for sh in SHAMS}) for st in ['S1','S2']}
 passes=support>=g['required_supporting_checkpoints'] and numerical and all(v['capture']>g['panel_each_stratum_checkpoint_median_capture_min_strict'] and all(e>0 for e in v['effects'].values()) for v in medians.values())
 return dict(supporting_checkpoints=support,total_checkpoints=3,numerical_valid=numerical,across_checkpoint_medians=medians,replicated=bool(passes),seed_replacement=False)
