"""File-independent privileged transport scientific engine, no Formal lifecycle."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from src.config import C
from src import kernel as k
from src.capture import capture
from src.core_decision import checkpoint_summary

def put(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def pack_npz(path,**arrays):
 np.savez_compressed(path,**arrays)
 with np.load(path,allow_pickle=False) as saved:
  assert set(saved.files)==set(arrays)
  for name,value in arrays.items():assert np.array_equal(saved[name],value),'SERIALIZATION_MISMATCH:'+name

@torch.no_grad()
def anchors(model,records):
 result=[]
 for key in ['factual_observations','counterfactual_observations']:
  obs=torch.as_tensor(np.asarray([r[key][:8] for r in records],np.float32));act=torch.as_tensor(np.asarray([r['actions'][:8] for r in records],np.float32));z,_=model.encode_history(obs,act);result.append(z)
 return result

def evaluate(model,records,U,checkpoint,registry,out):
 if U.shape!=(384,6) or not np.isfinite(U).all():raise ValueError('EXACT_U6_REQUIRED')
 if [r['intervention_unit_id'] for r in records]!=[u['unit_id'] for u in registry]:raise ValueError('RECORD_ORDER_IDENTITY_FAILURE')
 out=Path(out);out.mkdir(parents=True,exist_ok=True);before=k.statehash(model)
 for parameter in model.parameters():parameter.requires_grad_(False)
 f0,cf0=anchors(model,records);full=cf0-f0;u=torch.tensor(U,dtype=torch.float32);coeff=full@u;patch=coeff@u.T;o0=f0+patch
 actions=torch.tensor(np.asarray([r['actions'][7:19] for r in records],np.float32));assert actions.shape==(len(records),12,4) and torch.count_nonzero(actions)==0
 counts=dict(steps=0,observations=0,decodes=0,nonzero_actions=0);original_step=model.step;original_decode=model.decode
 def watched_step(z,a,observation=None):
  counts['steps']+=1;counts['observations']+=int(observation is not None);counts['nonzero_actions']+=int(torch.count_nonzero(a)!=0);return original_step(z,a,observation)
 def watched_decode(z):counts['decodes']+=1;return original_decode(z)
 model.step=watched_step;model.decode=watched_decode
 try:
  F,V,pred=k.factual_transport(model,f0,patch,u,actions)
  O,yO=k.ordinary(model,o0,actions);CF,yCF=k.ordinary(model,cf0,actions);plain,yF=k.ordinary(model,f0,actions);assert torch.equal(F,plain),'JVP_PRIMAL_DRIFT'
  matched=dict(F=F,O_PATCH=O,CF=CF,V=V,pred=pred,actions=actions,U=u,delta_full=full,oracle_coefficients=coeff,projected_anchor=patch,yF=yF,yO=yO,yCF=yCF)
  raw={name:value.detach().cpu().numpy() for name,value in matched.items()};pack_npz(out/'MATCHED.npz',**raw)
  outputs={};rows=[];cf=k.denormalize(raw['yCF'])
  for release_time in C['release_times']:
   cap=capture(V[release_time],O[release_time]-F[release_time]);errors={};valid=np.ones(len(records),bool);details={}
   for route in C['routes']:
    inserted=matched[route][release_time] if route!='T' else F[release_time]+pred[release_time]
    states,y=k.release(model,inserted,actions[:,release_time:]);yp=k.denormalize(y.detach().cpu().numpy());hp=states.detach().cpu().numpy()
    if route!='T':assert torch.equal(states,matched[route][release_time:]),'BASELINE_RELEASE_IDENTITY'
    errors[route]=k.rmse(yp[:,1:],cf[:,release_time+1:])
    okay=np.isfinite(hp).all(axis=(0,2))&np.isfinite(yp).all(axis=(1,2))&(np.linalg.norm(hp,axis=-1).max(axis=0)<C['engineering_limit'])&(np.abs(yp).max(axis=(1,2))<C['engineering_limit'])&(np.linalg.norm(yp,axis=-1).max(axis=1)<C['engineering_limit']);valid &=okay
    outputs[f'{route}_k{release_time}']=yp;outputs[f'{route}_insert_k{release_time}']=inserted.detach().cpu().numpy();outputs[f'{route}_next_k{release_time}']=hp[1]
    details[route]=dict(max_carrier_norm=np.linalg.norm(hp,axis=-1).max(axis=0).tolist(),max_abs_decode=np.abs(yp).max(axis=(1,2)).tolist(),position_range_flag=(np.abs(yp.reshape(len(records),-1,2,4)[...,:2]).max(axis=(1,2,3))>C['arena_position_limit']).tolist(),valid=okay.tolist())
   for i,unit in enumerate(registry):rows.append(dict(checkpoint=checkpoint,stratum=unit['stratum'],unit_id=unit['unit_id'],k=release_time,capture=float(cap[i]),errors={route:float(e[i]) for route,e in errors.items()},state_valid=bool(valid[i]),autonomy_valid=True,state_checks={route:{key:values[i] for key,values in detail.items()} for route,detail in details.items()}))
  expected=dict(steps=12*7+12*3+4*sum(12-t for t in C['release_times']),decodes=13*3+4*sum(13-t for t in C['release_times']),observations=0,nonzero_actions=0)
  assert counts==expected,counts
 finally:model.step=original_step;model.decode=original_decode
 assert k.statehash(model)==before
 pack_npz(out/'RELEASED.npz',**outputs);put(out/'UNIT_TIME_ROWS.json',rows);assert json.loads((out/'UNIT_TIME_ROWS.json').read_text())==rows
 summary=checkpoint_summary(rows,checkpoint,registry);put(out/'SUMMARY.json',summary)
 trace=dict(counts=counts,expected=expected,weights_before=before,weights_after=k.statehash(model),anchor_difference_rounding_max_abs=float(abs((o0-f0)-patch).max()),only_one_anchor_patch=True,transport_input_contract=['factual_anchor','projected_anchor','U6','future_actions'],later_O_or_CF_used_in_transport=False,serialized_arrays_verified=True)
 put(out/'TRACE.json',trace);return dict(summary=summary,trace=trace)
