"""Single monolithic standard LSTMCell; complete carrier is concatenated [h;c]."""
from __future__ import annotations
import torch
from torch import nn

class MonolithicLSTM(nn.Module):
 def __init__(self,hidden_size=192,encoder_width=224,decoder_width=256):
  super().__init__();self.hidden_size=hidden_size;self.carrier_dim=2*hidden_size
  self.observation_encoder=nn.Sequential(nn.Linear(12,encoder_width),nn.GELU(),nn.Linear(encoder_width,encoder_width),nn.GELU())
  self.recurrent=nn.LSTMCell(encoder_width+4,hidden_size)
  self.decoder=nn.Sequential(nn.Linear(hidden_size,decoder_width),nn.GELU(),nn.Linear(decoder_width,decoder_width),nn.GELU(),nn.Linear(decoder_width,8))
 def zero_carrier(self,batch,device,dtype):return torch.zeros(batch,self.carrier_dim,device=device,dtype=dtype)
 def unpack(self,z):
  if z.ndim!=2 or z.shape[-1]!=self.carrier_dim:raise ValueError('COMPLETE_H_AND_C_CARRIER_REQUIRED')
  return z[:,:self.hidden_size],z[:,self.hidden_size:]
 def pack(self,h,c):return torch.cat((h,c),dim=-1)
 def step(self,z,action,observation=None):
  h,c=self.unpack(z)
  if action.shape!=(z.shape[0],4):raise ValueError('FOUR_SCALAR_ACTION_INTERFACE_REQUIRED')
  if observation is None:
   # Known slot IDs preserved; no future sensor/oracle input.
   observation=z.new_zeros((z.shape[0],2,6));observation[:,1,5]=1
  if observation.shape!=(z.shape[0],2,6):raise ValueError('CANONICAL_OBSERVATION_SHAPE_REQUIRED')
  embedding=self.observation_encoder(observation.reshape(z.shape[0],12))
  hn,cn=self.recurrent(torch.cat((embedding,action),dim=-1),(h,c))
  return self.pack(hn,cn)
 def encode_history(self,observations,actions):
  z=self.zero_carrier(observations.shape[0],observations.device,observations.dtype);saved=[]
  for time_index in range(observations.shape[1]):
   z=self.step(z,actions[:,time_index],observations[:,time_index]);saved.append(z)
  return z,{'saved_carriers':saved,'carrier_complete':True}
 def transition(self,carrier,action):return self.step(carrier,action,None)
 def decode(self,carrier):
  h,_=self.unpack(carrier);return self.decoder(h)
