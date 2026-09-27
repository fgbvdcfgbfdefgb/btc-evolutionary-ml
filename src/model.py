import torch
from torch import nn
class Forecaster(nn.Module):
    def __init__(self,n_features,n_targets,hidden=256,layers=4,dropout=.1):
        super().__init__(); self.proj=nn.Linear(n_features,hidden)
        enc=nn.TransformerEncoderLayer(hidden,8,hidden*4,dropout,batch_first=True,norm_first=True,activation="gelu")
        self.encoder=nn.TransformerEncoder(enc,layers); self.norm=nn.LayerNorm(hidden); self.head=nn.Linear(hidden,n_targets)
    def forward(self,x): return self.head(self.norm(self.encoder(self.proj(x))[:,-1]))
