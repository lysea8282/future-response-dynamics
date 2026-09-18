"""Pure scientific adapter: canonical model, raw tangent JVP, Paper1 M3 weighting."""
import hashlib
from runtime_config import C
import numpy as np
import torch
def statehash(model):
    h=hashlib.sha256()
    for key,value in model.state_dict().items():h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest().upper()
def denormalize(y):
    return (np.asarray(y,dtype=np.float32)*np.asarray(C['normalizer_std'],dtype=np.float32)+np.asarray(C['normalizer_mean'],dtype=np.float32)).astype(np.float32)
def rmse(y,ref):
    # Inputs [N,time,8]; exclude no frames here, caller explicitly supplies remaining future.
    scale=np.asarray(C['primitive_scale'],dtype=np.float64).reshape(8)
    return np.sqrt(np.mean(((np.asarray(y,dtype=np.float64)-np.asarray(ref,dtype=np.float64))/scale)**2,axis=(1,2)))
def jvp(fn,h,v):
    return torch.autograd.functional.jvp(fn,h,v,create_graph=False,strict=False)
def factual_transport(model,factual_anchor,anchor_patch,U,actions):
    # No later P or CF argument, cache, closure, or access. Raw V is never orthonormalized here.
    h=factual_anchor;pred=anchor_patch;v=U.expand(len(h),-1,-1).clone()
    hs=[];vs=[];ps=[]
    for k in range(13):
        hs.append(h);vs.append(v);ps.append(pred)
        if k==12:break
        cols=torch.cat((v,pred[...,None]),dim=-1);new=[]
        with torch.no_grad():
            for j in range(5):
                nxt,tangent=jvp(lambda z:model.transition(z,actions[:,k]),h,cols[...,j].contiguous())
                new.append(tangent)
        h=nxt;v=torch.stack(new[:4],dim=-1);pred=new[4]
    return torch.stack(hs),torch.stack(vs),torch.stack(ps)
@torch.no_grad()
def ordinary(model,h,actions):
    states=[h];decoded=[model.decode(h)]
    for k in range(actions.shape[1]):
        h=model.transition(h,actions[:,k]);states.append(h);decoded.append(model.decode(h))
    return torch.stack(states),torch.stack(decoded,dim=1)
def matched(m,model,interface,records):
    dev=torch.device(C['device'])
    routes,_=m.construct_routes(model,records,interface,dev)
    actions=torch.as_tensor(np.asarray([r['actions'][7:19] for r in records]),dtype=torch.float32,device=dev)
    assert actions.shape==(len(records),12,4) and torch.count_nonzero(actions)==0
    U=torch.as_tensor(interface['U4'],dtype=torch.float32,device=dev)
    hf,v,pred=factual_transport(model,routes['F'],routes['P']-routes['F'],U,actions)
    hp,yp=ordinary(model,routes['P'],actions);hcf,ycf=ordinary(model,routes['CF'],actions)
    hf_plain,yf=ordinary(model,routes['F'],actions)
    assert torch.equal(hf,hf_plain),'JVP primal differs from ordinary F'
    return dict(F=hf,P=hp,CF=hcf,V=v,pred=pred,actions=actions,U=U,yF=yf,yP=yp,yCF=ycf)
def release(model,state,actions):
    # The inserted state is used once. No projector, observation or correction is available to this function.
    return ordinary(model,state,actions)

