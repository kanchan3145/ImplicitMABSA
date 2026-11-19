import torch
from torch import nn

class TextHead(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.LayerNorm(768),nn.Dropout(.2),nn.Linear(768,256),nn.GELU(),nn.Linear(256,3))
    def forward(self,x): return self.net(x['roberta'])

class ImageHead(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.LayerNorm(768),nn.Dropout(.2),nn.Linear(768,256),nn.GELU(),nn.Linear(256,3))
    def forward(self,x): return self.net(x['vit'])

class ConcatFusion(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,256),nn.GELU(),nn.Dropout(.2),nn.Linear(256,3))
    def forward(self,x):
        t=x['clip_text'][:,0];v=x['clip_image'][:,0]
        return self.net(torch.cat([t,v],dim=-1))

class CrossAttentionFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.t2v=nn.MultiheadAttention(512,8,dropout=.1,batch_first=True)
        self.v2t=nn.MultiheadAttention(512,8,dropout=.1,batch_first=True)
        self.norm_t=nn.LayerNorm(512);self.norm_v=nn.LayerNorm(512)
        self.net=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,256),nn.GELU(),nn.Dropout(.2),nn.Linear(256,3))
    def forward(self,x):
        t=x['clip_text'];v=x['clip_image'];mask=x['clip_mask'].bool()
        ta,_=self.t2v(t,v,v,need_weights=False)
        va,_=self.v2t(v,t,t,key_padding_mask=~mask,need_weights=False)
        t=self.norm_t(t+ta);v=self.norm_v(v+va)
        tp=(t*mask.unsqueeze(-1)).sum(1)/mask.sum(1).clamp(min=1).unsqueeze(-1)
        vp=v.mean(1)
        return self.net(torch.cat([tp,vp],-1))

MODELS={'text':TextHead,'image':ImageHead,'concat':ConcatFusion,'cross':CrossAttentionFusion}
