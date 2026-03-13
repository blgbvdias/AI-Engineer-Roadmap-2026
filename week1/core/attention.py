import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class CustomMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim//num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, past_key_value=None):
        B, T, C = x.size()
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        Q_h = Q.reshape(B, T, self.num_heads, self.head_dim).permute(0,2,1,3)   #B, num_heads, T, d
        K_h = K.reshape(B, T, self.num_heads, self.head_dim).permute(0,2,1,3)   #B, num_heads, T, d
        V_h = V.reshape(B, T, self.num_heads, self.head_dim).permute(0,2,1,3)   #B, num_heads, T, d

        if past_key_value is not None:
            past_k, past_v = past_key_value
            K_h = torch.cat([past_k, K_h], dim=2)
            V_h = torch.cat([past_v, V_h], dim=2)
        present_key_value = (K_h, V_h)

        scores = (Q_h@K_h.permute(0,1,3,2))/math.sqrt(self.head_dim)    #B, num_heads, T, T
        if T > 1:
            mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            scores = scores.masked_fill(mask==0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)   #B, H, T, T
        A = attention_weights@V_h   #B, H, T, D
        A = A.permute(0,2,1,3).contiguous().view(B, T, C)
        return self.out_proj(A), present_key_value
