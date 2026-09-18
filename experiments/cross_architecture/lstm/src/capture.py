import torch
from src.config import C
def capture(V,delta):
 left,s,_=torch.linalg.svd(V.double(),full_matrices=False)
 Q=left*(s>s[:,:1]*C['rank_rtol'])[:,None,:]
 d=delta.double();projection=(Q@(Q.transpose(1,2)@d[...,None])).squeeze(-1)
 return (projection.square().sum(-1)/(d.square().sum(-1)+C['eps'])).cpu().numpy()
