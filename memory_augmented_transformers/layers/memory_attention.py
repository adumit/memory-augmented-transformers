import math

from einops import rearrange
import torch
from torch import nn
import mlflow

from memorizing_transformers_pytorch.memorizing_transformers_pytorch import l2norm


class empty_context():
  def __enter__(self, *args, **kwargs):
    pass

  def __exit__(self, *args, **kwargs):
    pass


class KNNAttentionNoNewLayer(nn.Module):
  def __init__(
      self,
      num_retrieved_memories, 
      aggregate_after_ff=True, 
      do_not_grad_through_gpt_layers=False,
      normalize_qk=False
    ):
    super().__init__()

    self.num_retrieved_memories = num_retrieved_memories
    self.aggregate_after_ff = aggregate_after_ff
    self.do_not_grad_through_gpt_layers = do_not_grad_through_gpt_layers
    self.normalize_qk = normalize_qk

  def _split_memory_into_heads(self, tensor, num_heads, attn_head_size):
    return rearrange(tensor, 'b s m (h d) -> b h s m d', h=num_heads, d=attn_head_size)

  def memory_attn(self, query, mem_key, mem_value):
    # Memory key is dimension: (batch, n_heads, num_mems, head_dim)


    # (batch, heads, 1, seq, head_dim) * (batch, heads, mem, seq, head_dim)
    # Memory attn output: (batch, heads, mem, seq, seq)

    # Typical attn: (batch, heads, seq, head_dim) * (batch, heads, head_dim, seq)
    # Result in typical attn: (batch, heads, seq, seq)

    # Query is dimension: (batch, n_heads, 1, seq, head_dim)
    # Mem_key is dimension: (batch, n_heads, seq, head_dim, mem)
    # Result in mem attn_weights: (batch, n_heads, seq, seq, mem)
    attn_weights = torch.matmul(query.unsqueeze(-2), 
                                rearrange(mem_key, 'b h s m d -> b h s d m'))
    
    # V1:
    # (batch, n_heads, 1, seq, head_dim) * (batch, n_heads, num_mems, head_dim, seq)
    # Results in: (batch, n_heads, num_mems, seq, seq)
    # V2:
    # (batch, n_heads, 1, seq, head_dim) * (batch, n_heads, seq, head_dim, num_mems)
    # Results in: (batch, n_heads, seq, seq, num_mems)
    
    # TODO: I'm not sure about this normalization
    #  It seems maybe as useful as it is in the regular transformer, but unclear
    attn_weights = attn_weights / (mem_value.size(-1) ** 0.5)

    # I believe this should still be done over dim=-1 since the mems dimension
    # wouldn't affect this.
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    # Final dimension is going to be [batch, num_heads, seq_length, 1, dim_head]
    # So just squeeze out the -2 dimension
    attn_output = torch.matmul(attn_weights, mem_value).squeeze(-2)

    return attn_output, attn_weights
  
  def forward(self, previous_hidden, cur_layer, knn_memory, attention_mask, head_mask, step=None, memory_per_head=False):
    grad_context = torch.no_grad if self.do_not_grad_through_gpt_layers else empty_context

    with grad_context():
      residual = previous_hidden
      prev_hidden_sum = float(torch.sum(previous_hidden.clone().detach().cpu()))

      hidden_states = cur_layer.ln_1(residual)

      attention_layer = cur_layer.attn

      query, key, value = attention_layer.c_attn(hidden_states).split(attention_layer.split_size, dim=2)
      query = attention_layer._split_heads(query, attention_layer.num_heads, attention_layer.head_dim)
      key = attention_layer._split_heads(key, attention_layer.num_heads, attention_layer.head_dim)
      value = attention_layer._split_heads(value, attention_layer.num_heads, attention_layer.head_dim)
      
      if self.normalize_qk:
        query, key = map(l2norm, (query, key))

      if memory_per_head:
        mem_ks = []
        mem_vs = []
        for head in range(query.shape[1]):
          search_query = query[:, head, :, :]
          mem_kv, mem_mask = knn_memory[head].search(search_query, self.num_retrieved_memories)
          mem_k_for_head, mem_v_for_head = mem_kv.unbind(dim = -2)
          mem_ks.append(mem_k_for_head)
          mem_vs.append(mem_v_for_head)
        mem_k_split_into_heads = torch.stack(mem_ks, dim=1)
        mem_v_split_into_heads = torch.stack(mem_vs, dim=1)
      else:
        search_query = rearrange(query, 'b h s d -> b s (h d)')
        mem_kv, mem_mask = knn_memory.search(search_query, self.num_retrieved_memories)
        mem_k, mem_v = mem_kv.unbind(dim = -2)
        mem_k_split_into_heads = self._split_memory_into_heads(mem_k, attention_layer.num_heads, attention_layer.head_dim)
        mem_v_split_into_heads = self._split_memory_into_heads(mem_v, attention_layer.num_heads, attention_layer.head_dim)

      if memory_per_head:
        for head in range(query.shape[1]):
          key_to_insert = key[:, head, :, :]
          value_to_insert = value[:, head, :, :]
          new_kv_mem = torch.stack((key_to_insert, value_to_insert), dim=-2).detach()
          if new_kv_mem.numel() > 0:
            knn_memory[head].add(new_kv_mem)
      else:
        key_to_insert = rearrange(key, 'b h s d -> b s (h d)')
        value_to_insert = rearrange(value, 'b h s d -> b s (h d)')
        new_kv_memories = torch.stack((key_to_insert, value_to_insert), dim = -2).detach()

        if new_kv_memories.numel() > 0:
          knn_memory.add(new_kv_memories)

      attn_output, attn_weights = self.memory_attn(query, mem_k_split_into_heads, mem_v_split_into_heads)

      attn_output = attention_layer._merge_heads(attn_output, attention_layer.num_heads, attention_layer.head_dim)

      attn_output = attention_layer.c_proj(attn_output)
      attn_output = attention_layer.resid_dropout(attn_output)

      hidden_states = attn_output + residual
      residual = hidden_states

      hidden_states = cur_layer.ln_2(hidden_states)
      # Currently outputs [batch, seq, num_heads*dim_head]
      feed_forward_hidden_states = cur_layer.mlp(hidden_states)
      hidden_states = residual + feed_forward_hidden_states

      knn_hidden_sum = float(torch.sum(hidden_states.clone().detach().cpu()))
      mlflow.log_metrics({
        "gpt-2-sum": prev_hidden_sum,
        "memory_computed-sum": knn_hidden_sum 
      }, step=step)

    return hidden_states


class KNNAttentionAggBeforeMLP(nn.Module):
  def __init__(
      self,
      num_retrieved_memories,
      normalize_qk=False,
      apply_linear_g=True
    ):
    super().__init__()

    self.num_retrieved_memories = num_retrieved_memories
    self.apply_linear_g = apply_linear_g
    self.normalize_qk = normalize_qk

  def _split_memory_into_heads(self, tensor, num_heads, attn_head_size):
    return rearrange(tensor, 'b s m (h d) -> b h s m d', h=num_heads, d=attn_head_size)

  def memory_attn(self, query, mem_key, mem_value):
    # Memory key is dimension: (batch, n_heads, num_mems, head_dim)

    # (batch, heads, 1, seq, head_dim) * (batch, heads, mem, seq, head_dim)
    # Memory attn output: (batch, heads, mem, seq, seq)

    # Typical attn: (batch, heads, seq, head_dim) * (batch, heads, head_dim, seq)
    # Result in typical attn: (batch, heads, seq, seq)

    # Query is dimension: (batch, n_heads, 1, seq, head_dim)
    # Mem_key is dimension: (batch, n_heads, seq, head_dim, mem)
    # Result in mem attn_weights: (batch, n_heads, seq, seq, mem)
    attn_weights = torch.matmul(query.unsqueeze(-2), 
                                rearrange(mem_key, 'b h s m d -> b h s d m'))
    
    # TODO: I'm not sure about this normalization
    #  It seems maybe as useful as it is in the regular transformer, but unclear
    attn_weights = attn_weights / (mem_value.size(-1) ** 0.5)

    # I believe this should still be done over dim=-1 since the mems dimension
    # wouldn't affect this.
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    # Final dimension is going to be [batch, num_heads, seq_length, 1, dim_head]
    # So just squeeze out the -2 dimension
    attn_output = torch.matmul(attn_weights, mem_value).squeeze(-2)

    return attn_output, attn_weights
  
  def forward(self, previous_hidden, cur_layer, knn_memory, attention_mask, head_mask, g_val, step=None, memory_per_head=False):
    residual = previous_hidden
    prev_hidden_sum = float(torch.sum(previous_hidden.clone().detach().cpu()))

    hidden_states = cur_layer.ln_1(residual)

    attention_layer = cur_layer.attn

    device = hidden_states.device
    query, key, value = attention_layer.c_attn(hidden_states).split(attention_layer.split_size, dim=2)
    query = attention_layer._split_heads(query, attention_layer.num_heads, attention_layer.head_dim)
    key = attention_layer._split_heads(key, attention_layer.num_heads, attention_layer.head_dim)
    value = attention_layer._split_heads(value, attention_layer.num_heads, attention_layer.head_dim)
    
    if self.normalize_qk:
      query, key = map(l2norm, (query, key))

    if memory_per_head:
      mem_ks = []
      mem_vs = []
      for head in range(query.shape[1]):
        search_query = query[:, head, :, :]
        mem_kv, mem_mask = knn_memory[head].search(search_query, self.num_retrieved_memories)
        mem_k_for_head, mem_v_for_head = mem_kv.unbind(dim = -2)
        mem_ks.append(mem_k_for_head)
        mem_vs.append(mem_v_for_head)
      mem_k_split_into_heads = torch.stack(mem_ks, dim=1)
      mem_v_split_into_heads = torch.stack(mem_vs, dim=1)

      for head in range(query.shape[1]):
        key_to_insert = key[:, head, :, :]
        value_to_insert = value[:, head, :, :]
        new_kv_mem = torch.stack((key_to_insert, value_to_insert), dim=-2).detach()
        if new_kv_mem.numel() > 0:
          knn_memory[head].add(new_kv_mem)
    else:
      search_query = rearrange(query, 'b h s d -> b s (h d)')
      mem_kv, mem_mask = knn_memory.search(search_query, self.num_retrieved_memories)
      mem_k, mem_v = mem_kv.unbind(dim = -2)
      mem_k_split_into_heads = self._split_memory_into_heads(mem_k, attention_layer.num_heads, attention_layer.head_dim)
      mem_v_split_into_heads = self._split_memory_into_heads(mem_v, attention_layer.num_heads, attention_layer.head_dim)

      key_to_insert = rearrange(key, 'b h s d -> b s (h d)')
      value_to_insert = rearrange(value, 'b h s d -> b s (h d)')
      new_kv_memories = torch.stack((key_to_insert, value_to_insert), dim = -2).detach()

      if new_kv_memories.numel() > 0:
        knn_memory.add(new_kv_memories)

    # This replicates the output from the GPT-2 attention block
    # It includes: attention computation, c_projection, and residual dropout
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L299
    mem_attn_output, mem_attn_weights = self.memory_attn(query, mem_k_split_into_heads, mem_v_split_into_heads)
    mem_attn_output = attention_layer._merge_heads(mem_attn_output, attention_layer.num_heads, attention_layer.head_dim)
    attn_output = attention_layer.c_proj(mem_attn_output)
    attn_output = attention_layer.resid_dropout(attn_output)

    std_attn_output = cur_layer.attn(
        hidden_states,
        attention_mask=attention_mask,
        head_mask=head_mask
    )
    std_attn_output = std_attn_output[0]
    
    # Add the mem and standard attention before adding residual and 
    # pass through to MLP
    g_val_per_head = False
    if g_val.shape[0] != 1:
      std_attn_output = attention_layer._split_heads(std_attn_output, attention_layer.num_heads, attention_layer.head_dim)
      mem_attn_output = attention_layer._split_heads(mem_attn_output, attention_layer.num_heads, attention_layer.head_dim)
      # Turn g-value in a vector that we multiply along the head dimension
      g_val = g_val.view(1, -1, 1, 1)
      # Mark this here because down below we want to check whether this block
      # was active
      g_val_per_head = True

    if self.apply_linear_g:
      attn_output = ((1 - g_val)*std_attn_output + g_val*mem_attn_output)
    else:
      attn_output = (std_attn_output + g_val*mem_attn_output)

    if g_val_per_head:
      attn_output = attention_layer._merge_heads(attn_output, attention_layer.num_heads, attention_layer.head_dim)

    hidden_states = attn_output + residual
    residual = hidden_states

    hidden_states = cur_layer.ln_2(hidden_states)
    # Currently outputs [batch, seq, num_heads*dim_head]
    feed_forward_hidden_states = cur_layer.mlp(hidden_states)
    hidden_states = residual + feed_forward_hidden_states

    knn_hidden_sum = float(torch.sum(hidden_states.clone().detach().cpu()))
    mlflow.log_metrics({
       "gpt-2-sum": prev_hidden_sum,
       "memory_computed-sum": knn_hidden_sum 
    }, step=step)

    return hidden_states