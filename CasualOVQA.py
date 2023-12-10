# Structure of the CIQNet

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F



class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class SIE(nn.Module):
    def __init__(self):
        super().__init__()
        self.t1=nn.Conv1d(512,64,3,padding=1)
        self.t3 = nn.Conv1d(64, 512, 3, padding=1)
    def forward(self,x):

        x=x.transpose(1, 2)
        x0 = x
        x=self.t1(x)
        x=F.relu(x,True)
        x=self.t3(x)
        x = F.relu(x, True)
        # x=torch.cat((x,x0),dim=1)
        x=x+x0
        return x

class Quality_Transformer(nn.Module):
    def __init__(self, dropout = 0.1, emb_dropout = 0.1):
        super().__init__()

        self.quality_token = nn.Parameter(torch.randn(1, 1, 512))#1024
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(512, 3, 8, 128, 128, dropout)


        self.mlp_head = nn.Sequential(
            nn.LayerNorm(512),#1024
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = x.transpose(1,2)

        quality_tokens =self.quality_token.expand(x.shape[0], -1 ,-1)
        x = torch.cat((quality_tokens, x), dim=1)
        x = self.transformer(x)
        x=x[:, 0]
        return self.mlp_head(x).view(-1)

class CasualVQA(nn.Module):
    def __init__(self):
        super(CasualVQA, self).__init__()
        self.weighting_net=nn.Sequential(nn.Linear(513,64),nn.Dropout(0.2),nn.ReLU(),nn.Linear(64,1),nn.Softmax(dim=-2))
        self.qit=Quality_Transformer()
        self.sie=SIE()
    def forward(self,v):
        coding = torch.randn((v.shape[0], v.shape[1], 5, 1), dtype=torch.float, device=torch.device('cuda'))
        coding[:, :, 0, :] =3.14159/5*2
        coding[:, :, 1, :] = 3.14159/5
        coding[:, :, 2, :] = 0
        coding[:, :, 3, :] = 3.14159/5
        coding[:, :, 4, :] = 3.14159/5*2
        v2=torch.cat((v,coding),dim=-1)

        w=self.weighting_net(v2)
        v=v*w
        v=torch.sum(v,dim=-2)

        v=self.sie(v)

        q=self.qit(v)

        return q.view(-1)
