import math

from einops import rearrange
import torch
from torch import nn, einsum

from memorizing_transformers_pytorch.memorizing_transformers_pytorch import l2norm


class KNNAttentionSoftmaxOverLocalAndDistant(nn.Module):
    def __init__(
        self, 
        heads, 
        dim_head,
        num_retrieved_memories,
        normalize_qk=False,
        apply_linear_g=True,
        normalize_query_for_attn_mult=False,
        include_scale_parameter=False,
        scale_param=20.,
    ):
        super().__init__()

        self.n_heads = heads
        self.dim_head = dim_head
        self.num_retrieved_memories = num_retrieved_memories
        self.apply_linear_g = apply_linear_g
        self.normalize_qk = normalize_qk
        self.normalize_query_for_attn_mult = normalize_query_for_attn_mult

        self.include_scale_parameter = include_scale_parameter
        if self.include_scale_parameter:
            self.scale = nn.Parameter(torch.ones(heads, 1, 1) * math.log(scale_param))


    def _split_memory_into_heads(self, tensor, num_heads, attn_head_size):
        return rearrange(tensor, 'b s m (h d) -> b h s m d', h=num_heads, d=attn_head_size)
  
    def forward(self, previous_hidden, mem_layer, knn_memory, attention_mask, head_mask, g_val, step=None, memory_per_head=False, standard_layer=None):
        residual = previous_hidden
        hidden_states = mem_layer.ln_1(residual)
        attention_layer = mem_layer.attn

        query, key, value = attention_layer.c_attn(hidden_states).split(attention_layer.split_size, dim=2)
        query = attention_layer._split_heads(query, attention_layer.num_heads, attention_layer.head_dim)
        key = attention_layer._split_heads(key, attention_layer.num_heads, attention_layer.head_dim)
        value = attention_layer._split_heads(value, attention_layer.num_heads, attention_layer.head_dim)

        if memory_per_head:
            mem_ks = []
            mem_vs = []
            for head in range(query.shape[1]):
                search_query = query[:, head, :, :]
                if self.normalize_qk:
                    search_query = l2norm(search_query)
                mem_kv, _ = knn_memory[head].search(search_query, self.num_retrieved_memories)
                mem_k_for_head, mem_v_for_head = mem_kv.unbind(dim = -2)
                mem_ks.append(mem_k_for_head)
                mem_vs.append(mem_v_for_head)
            mem_k_split_into_heads = torch.stack(mem_ks, dim=1)
            mem_v_split_into_heads = torch.stack(mem_vs, dim=1)

            for head in range(query.shape[1]):
                key_to_insert = key[:, head, :, :]
                if self.normalize_qk:
                    key_to_insert = l2norm(key_to_insert)
                value_to_insert = value[:, head, :, :]
                new_kv_mem = torch.stack((key_to_insert, value_to_insert), dim=-2).detach()
                if new_kv_mem.numel() > 0:
                    knn_memory[head].add(new_kv_mem)
        else:
            search_query = rearrange(query, 'b h s d -> b s (h d)')
            search_query = l2norm(search_query)
            mem_kv, _ = knn_memory.search(search_query, self.num_retrieved_memories)
            mem_k, mem_v = mem_kv.unbind(dim = -2)
            mem_k_split_into_heads = self._split_memory_into_heads(mem_k, attention_layer.num_heads, attention_layer.head_dim)
            mem_v_split_into_heads = self._split_memory_into_heads(mem_v, attention_layer.num_heads, attention_layer.head_dim)

            key_to_insert = rearrange(key, 'b h s d -> b s (h d)')
            if self.normalize_qk:
                key_to_insert = l2norm(key_to_insert)
            value_to_insert = rearrange(value, 'b h s d -> b s (h d)')
            new_kv_memories = torch.stack((key_to_insert, value_to_insert), dim = -2).detach()

            if new_kv_memories.numel() > 0:
                knn_memory.add(new_kv_memories)

        # Handle memory attention up to softmax
        if self.normalize_query_for_attn_mult:
            query_for_mem_attn = l2norm(query)
        else:
            query_for_mem_attn = query
        mem_attn_weights = einsum('b h i d, b h i j d -> b h i j', query_for_mem_attn, mem_k_split_into_heads)
        mem_attn_weights = mem_attn_weights / (mem_v_split_into_heads.size(-1) ** 0.5)
        if self.include_scale_parameter:
            scale = self.scale.exp()
            mem_attn_weights = mem_attn_weights * scale

        if standard_layer is not None:
            std_attn_layer = standard_layer.attn
            hidden_states_standard = standard_layer.ln_1(residual)
            query_std, key_std, value_std = std_attn_layer.c_attn(hidden_states_standard).split(std_attn_layer.split_size, dim=2)
            query_std = std_attn_layer._split_heads(query_std, std_attn_layer.num_heads, std_attn_layer.head_dim)
            key_std = std_attn_layer._split_heads(key_std, std_attn_layer.num_heads, std_attn_layer.head_dim)
            value_std = std_attn_layer._split_heads(value_std, std_attn_layer.num_heads, std_attn_layer.head_dim)
            std_attn_weights = torch.matmul(query_std, key_std.transpose(-1, -2))
            std_attn_weights = std_attn_weights / (value_std.size(-1) ** 0.5)
        else:
            std_attn_weights = torch.matmul(query, key.transpose(-1, -2))
            std_attn_weights = std_attn_weights / (value.size(-1) ** 0.5)
            value_std = value
    
        #####
        # Implement the causal mask for the standard memory
        #####
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = attention_layer.bias[:, :, key_length - query_length : key_length, :key_length].to(torch.bool)
        mask_value = torch.finfo(std_attn_weights.dtype).min
        mask_value = torch.tensor(mask_value, dtype=std_attn_weights.dtype).to(std_attn_weights.device)
        std_attn_weights = torch.where(causal_mask, std_attn_weights, mask_value)

        all_attn_weights = torch.cat((mem_attn_weights, std_attn_weights), dim = -1)

        all_attn_weights = nn.functional.softmax(all_attn_weights, dim=-1)

        local_attn, mem_attn = all_attn_weights[..., self.num_retrieved_memories:], all_attn_weights[..., :self.num_retrieved_memories]
        local_out = torch.matmul(local_attn, value_std)

        mem_out = einsum('b h i j, b h i j d -> b h i d', mem_attn, mem_v_split_into_heads)

        g_val = g_val.view(1, -1, 1, 1)
        attn_output = ((1 - g_val)*local_out + g_val*mem_out)
        attn_output = attention_layer._merge_heads(attn_output, attention_layer.num_heads, attention_layer.head_dim)
        attn_output = attention_layer.c_proj(attn_output)
        attn_output = attention_layer.resid_dropout(attn_output)

        hidden_states = attn_output + residual
        residual = hidden_states

        hidden_states = mem_layer.ln_2(hidden_states)
        # Currently outputs [batch, seq, num_heads*dim_head]
        feed_forward_hidden_states = mem_layer.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        return hidden_states