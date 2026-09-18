"""Outcome-independent semantic sham construction; no later P/CF input."""
import copy,hashlib
import numpy as np
import torch
from runtime_config import C
from sham_kernel import jvp
def anchor_controls(m,interface,records,actual_delta,checkpoint,units):
    wrong=[];meta=[]
    for r in records:
        obj=1-int(r['edit_object_id']);axis=int(r['edit_axis'])
        dv=float(r['counterfactual_values'][0])-float(r['factual_values'][0])
        baseline=float(r['factual_state_t'][obj][2+axis])
        q=dict(edit_object_id=obj,edit_axis=axis,factual_values=[baseline],counterfactual_values=[baseline+dv],factual_state_t=copy.deepcopy(r['factual_state_t']))
        wrong.append(q)
        assert abs((q['counterfactual_values'][0]-q['factual_values'][0])-dv)<1e-12
        meta.append(dict(original_object=int(r['edit_object_id']),wrong_object=obj,axis=axis,signed_delta=dv,wrong_baseline=baseline,wrong_requested_value=baseline+dv))
    wo=m.patch_delta(wrong,interface)
    x=m.primitive_features(records);x[:,1:]=(x[:,1:]-interface['feature_mean'][1:])/interface['feature_scale'][1:];x[:,0]=1
    coeff=x@interface['affine_weights'];U=interface['U4'].astype(np.float64)
    direction=[];norm=np.linalg.norm(np.asarray(actual_delta,dtype=np.float64),axis=1)
    for i,(c,u) in enumerate(zip(coeff,units)):
        literal=C['inplane_seed_literal'].format(checkpoint=checkpoint,unit_id=u['unit_id'])
        digest=hashlib.sha256(literal.encode('utf-8')).digest()
        seed=int.from_bytes(digest[:8],'big')
        g=np.random.Generator(np.random.PCG64(seed)).standard_normal(4)
        cc=float(c@c)
        if cc<=C['eps']:raise RuntimeError('INPLANE_COEFFICIENT_NOT_EVALUABLE')
        s=g-c*((g@c)/cc);fallback=np.linalg.norm(s)<=1e-12
        if fallback:
            g=np.eye(4)[np.argmin(np.abs(c))]
            s=g-c*((g@c)/cc)
        s=s/np.linalg.norm(s)
        assert abs(s@c)/np.linalg.norm(c)<C['coefficient_orthogonality_tolerance']
        direction.append(s)
        meta[i].update(seed_literal=literal,seed=seed,seed_sha256=digest.hex().upper(),fallback=bool(fallback),coefficient_relative_dot=float(abs(s@c)/np.linalg.norm(c)))
    direction=np.array(direction);ip=((direction@U.T)*norm[:,None]).astype(np.float32)
    residual=np.linalg.norm(np.asarray(actual_delta,dtype=np.float64)-coeff@U.T,axis=1)/(norm+C['eps'])
    assert residual.max()<C['anchor_relative_tolerance']
    assert np.max(np.abs(U.T@U-np.eye(4)))<1e-6
    assert np.max(np.abs(np.linalg.norm(ip.astype(np.float64),axis=1)-norm)/(norm+C['eps']))<C['norm_relative_tolerance']
    return wo,ip,coeff,direction,meta,residual
def transport_vectors(model,factual_states,actions,anchor_vectors):
    """A raw JVP recurrence along supplied factual path, with no patched-path argument."""
    v=anchor_vectors;out=[v]
    for k in range(12):
        with torch.no_grad():
            primal,v=jvp(lambda h:model.transition(h,actions[:,k]),factual_states[k],v)
        assert torch.equal(primal,factual_states[k+1])
        out.append(v)
    return torch.stack(out)
def norm_match(raw,target):
    rn=torch.linalg.vector_norm(raw.double(),dim=-1)
    tn=torch.linalg.vector_norm(target.double(),dim=-1)
    scale=tn/(rn+C['eps'])
    vector=(raw.double()*scale[:,None]).float()
    if torch.any(rn<=C['eps']):raise RuntimeError('SHAM_TRANSPORT_NORM_NOT_EVALUABLE')
    relative=torch.abs(torch.linalg.vector_norm(vector.double(),dim=-1)-tn)/(tn+C['eps'])
    assert float(relative.max())<C['norm_relative_tolerance']
    return vector,rn,tn,scale,relative

