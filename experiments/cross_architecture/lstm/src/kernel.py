import hashlib
import numpy as np
import torch
from src.config import C
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
            for j in range(U.shape[1]+1):
                nxt,tangent=jvp(lambda z:model.transition(z,actions[:,k]),h,cols[...,j].contiguous())
                new.append(tangent)
        h=nxt;v=torch.stack(new[:U.shape[1]],dim=-1);pred=new[U.shape[1]]
    return torch.stack(hs),torch.stack(vs),torch.stack(ps)

def ordinary(model,h,actions):
    states=[h];decoded=[model.decode(h)]
    for k in range(actions.shape[1]):
        h=model.transition(h,actions[:,k]);states.append(h);decoded.append(model.decode(h))
    return torch.stack(states),torch.stack(decoded,dim=1)

def release(model,state,actions):
    # The inserted state is used once. No projector, observation or correction is available to this function.
    return ordinary(model,state,actions)
