import math

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange


def l2norm(t):
    return F.normalize(t, dim = -1)


def exists(val):
    return val is not None


def stable_softmax(t, dim = -1):
    t = t - t.amax(dim = dim, keepdim = True).detach()
    return F.softmax(t, dim = dim)


class KNNAttentionOptionalLocal(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        num_retrieved_memories = 32,
        attn_scale_init = 20,
        only_memory_attn = False,
    ):
        super().__init__()
        self.heads = heads
        self.scale = nn.Parameter(torch.ones(heads, 1, 1) * math.log(attn_scale_init))

        inner_dim = heads * dim_head
        self.num_retrieved_memories = num_retrieved_memories

        self.dropout = nn.Dropout(dropout)
        self.knn_mem_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)
        
        self.only_memory_attn = only_memory_attn
            

    def forward(
        self,
        x,
        *,
        knn_memory,
        add_knn_memory = True,
        rel_pos_bias = None
    ):
        b, n, h, device = *x.shape[:2], self.heads, x.device
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        # in paper, they showed normalizing of keys led to more stable training
        # we'll just go with full cosine sim attention https://arxiv.org/abs/2010.04245

        q = l2norm(q)

        # Optionally, calculate local attention

        scale = self.scale.exp()

        # calculate knn attention over memory, if index is passed in

        mem_kv, mem_mask = knn_memory.search(q, self.num_retrieved_memories)
        mem_k, mem_v = mem_kv.unbind(dim = -2)

        sim_mem = einsum('b h i d, b h i j d -> b h i j', q, mem_k) * scale
        mask_value = -torch.finfo(sim_mem.dtype).max
        sim_mem = sim_mem.masked_fill(~mem_mask, mask_value)

        # calculate new XL memories, as well as memories to be discarded
        
        new_kv_memories = torch.stack((k, v), dim = -2).detach()

        new_kv_memories_discarded = new_kv_memories

        # add memories to be discarded into KNN memory

        if add_knn_memory and new_kv_memories_discarded.numel() > 0:
            knn_memory.add(new_kv_memories_discarded)

        # attention (combining local and distant)
        
        if self.only_memory_attn:
            sim = sim_mem
        
        attn = stable_softmax(sim)
        attn = self.dropout(attn)
        
        if self.only_memory_attn:
            mem_attn = attn
            mem_out = einsum('b h i j, b h i j d -> b h i d', mem_attn, mem_v)
            out = mem_out

        # combine heads and project out

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), None