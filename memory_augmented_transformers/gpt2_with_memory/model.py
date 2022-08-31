from typing import Optional, Tuple
from contextlib import contextmanager
from pathlib import Path
from copy import deepcopy

import torch
from torch import nn
import transformers

from memory_augmented_transformers.layers.memory_attention import KNNAttentionSoftmaxOverLocalAndDistant
from memory_augmented_transformers.knn_memory.memories import KNNMemoryListCustPerHead, KNNMemoryListCust


def exists(val):
    return val is not None



class GPT2WithMemory(nn.Module):
    def __init__(
        self, 
        model, 
        memory_layer_inds=(3,), 
        clear_memories_on_eos_token_id=None,
        knn_mem_dir="./memory/", 
        max_mems=16384, 
        num_mems_retrieved=4,
        allow_body_finetuning=False,
        use_sigmoid_for_g=False,
        apply_linear_g=True,
        use_tanh_for_g=False,
        use_softmax_over_localdistant_layer=False,
        use_knn_mems_per_head=False,
        g_per_head=False,
        normalize_qk=False,
        normalize_query_for_attn_mult=False,
        include_scale_parameter=False,
        create_memory_gpt_layer_copy=False
    ):
        super().__init__()
        self.device = model.device
        
        self.model = model
        if allow_body_finetuning is False:
            for param in self.model.parameters():
                param.requires_grad = False

        heads = self.model.config.n_head
        dim_head = int(self.model.config.n_embd / heads)
        assert self.model.config.n_embd / heads == dim_head

        self.use_knn_mems_per_head = use_knn_mems_per_head
        self.use_softmax_over_localdistant_layer = use_softmax_over_localdistant_layer
        self.g_per_head = g_per_head
        self.normalize_qk = normalize_qk
        self.dim_head = dim_head
        self.use_sigmoid_for_g = use_sigmoid_for_g
        self.use_tanh_for_g = use_tanh_for_g
        self.clear_memories_on_eos_token_id = clear_memories_on_eos_token_id
        self.knn_memories_directory = knn_mem_dir
        self.memory_layer_inds = memory_layer_inds
        self.normalize_query_for_attn_mult = normalize_query_for_attn_mult
        self.include_scale_parameter = include_scale_parameter
        self.create_memory_gpt_layer_copy = create_memory_gpt_layer_copy
        self.mem_layer_ind_to_attn_ind = {
            mem_layer_ind: attn_ind 
            for attn_ind, mem_layer_ind in enumerate(memory_layer_inds)
        }

        self.num_memory_layers = len(memory_layer_inds)
        self.apply_linear_g = apply_linear_g
        mem_dim = dim_head if self.use_knn_mems_per_head else dim_head*heads
        self.knn_mem_kwargs = dict(
            dim=mem_dim,
            max_memories=max_mems,
            multiprocessing=False
        )
        self.config = self.model.config
        self.dtype = self.model.dtype

        if self.use_softmax_over_localdistant_layer:
            assert self.g_per_head is True
            self.knn_attns = [
                KNNAttentionSoftmaxOverLocalAndDistant(
                    dim_head=dim_head,
                    heads=heads,
                    num_retrieved_memories=num_mems_retrieved,
                    apply_linear_g=apply_linear_g,
                    normalize_qk=normalize_qk,
                    normalize_query_for_attn_mult=normalize_query_for_attn_mult,
                    include_scale_parameter=include_scale_parameter,
                ).to(self.device) for _ in memory_layer_inds]
        else:
            self.knn_attns = []

        if self.create_memory_gpt_layer_copy:
            self.memory_attention_layers = [
                deepcopy(self.model.h[layer_ind])
                for layer_ind in memory_layer_inds
            ]
            for i, mem_attn_layer in enumerate(self.memory_attention_layers):
                setattr(self, f"mem_attn_layer-{i}", mem_attn_layer)
          
        if self.g_per_head:
            self.gs = [torch.nn.Parameter(data=torch.zeros(self.model.config.n_head), requires_grad=True) 
                       for _ in memory_layer_inds]
            for i, g in enumerate(self.gs):
                setattr(self, f"g-{i}", g)
        else:
            self.gs = [torch.nn.Parameter(data=torch.zeros(1), requires_grad=True) 
                       for _ in memory_layer_inds]
            for i, g in enumerate(self.gs):
                setattr(self, f"g-{i}", g)

        for i, knn_attn in enumerate(self.knn_attns):
            setattr(self, f"knn_attn-{i}", knn_attn)

    def freeze_body(self):
        for param in self.model.parameters():
            param.requires_grad = False
    
    def unfreeze_body(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def get_body_parameters(self):
        return self.model.parameters()

    def get_knn_parameters(self):
        for knn_attn in self.knn_attns:
            for param in knn_attn.parameters():
                yield param
        for g in self.gs:
            yield g
        if hasattr(self, "memory_attention_layers"):
            for mem_attn_layer in self.memory_attention_layers:
                for param in mem_attn_layer.parameters():
                    yield param
        
    def create_knn_memories(
        self,
        *,
        batch_size
    ):
        if self.use_knn_mems_per_head:
            return KNNMemoryListCustPerHead.create_memories(
                batch_size = batch_size,
                num_memory_layers = self.num_memory_layers,
                memories_directory = self.knn_memories_directory,
                num_heads=self.model.config.n_head
            )(**self.knn_mem_kwargs)
        else:
            return KNNMemoryListCust.create_memories(
                batch_size = batch_size,
                num_memory_layers = self.num_memory_layers,
                memories_directory = self.knn_memories_directory,
            )(**self.knn_mem_kwargs)

    @contextmanager
    def knn_memories_context(
        self,
        **kwargs
    ):
        knn_dir = Path(self.knn_memories_directory)
        knn_dir.mkdir(exist_ok = True, parents = True)
        lock = FileLock(str(knn_dir / 'mutex'))

        with lock:
            knn_memories = self.create_knn_memories(**kwargs)
            yield knn_memories
            if self.use_knn_mems_per_head:
                for i in range(len(self.gs)):
                    knn_memories[i].cleanup()
            else:
                knn_memories.cleanup()
            
    def clear_memory(self, x, token_id, knn_memories, step=None):
        """ clears the KNN memories based on if the batch row contains the specified token id """
        """ for auto-clearing KNN memories based on start and end of strings """

        clear_memory = (x == token_id).any(dim = -1)
        batch_indices, *_ = clear_memory.nonzero(as_tuple = True)
        batch_indices_to_clear = batch_indices.tolist()

        if self.use_knn_mems_per_head:
            for i in range(len(self.gs)):
                knn_memories[i].clear_memory(batch_indices_to_clear)
        else:
            knn_memories.clear_memory(batch_indices_to_clear)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        knn_memories = None,
        step = None
    ) -> Tuple:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.model.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.model.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.model.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.model.wte(input_ids)
        position_embeds = self.model.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.model.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.model.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        prev_hidden = None
        for i, (block, layer_past) in enumerate(zip(self.model.h, past_key_values)):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            prev_hidden = hidden_states.clone()

            if i in self.memory_layer_inds and knn_memories is not None:
                ind_to_get = self.mem_layer_ind_to_attn_ind[i]
                if step is not None and (step+1) % 100 == 0:
                    # Ensure that we're actually getting some signal from our memory
                    # I added this in because I was observing errors with the parallel
                    # processing where it appeared the delayed functions weren't 
                    # getting called
                    assert any(any(knn.is_trained for knn in knn_mem.knns) for knn_mem in knn_memories[ind_to_get])

                g_val = self.gs[ind_to_get]
                if self.use_sigmoid_for_g:
                    g_val = torch.nn.functional.sigmoid(g_val)
                elif self.use_tanh_for_g:
                    g_val = torch.tanh(g_val)

                knn_layer = self.knn_attns[ind_to_get]
                hidden_states = knn_layer(
                    previous_hidden=prev_hidden,
                    mem_layer=self.memory_attention_layers[ind_to_get] if hasattr(self, 'memory_attention_layers') else block,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    knn_memory=knn_memories[ind_to_get],
                    step=step,
                    g_val=g_val,
                    memory_per_head=self.use_knn_mems_per_head,
                    standard_layer=block if hasattr(self, 'memory_attention_layers') else None
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                hidden_states = outputs[0]

        hidden_states = self.model.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        if exists(self.clear_memories_on_eos_token_id) and knn_memories is not None:
            self.clear_memory(input_ids, self.clear_memories_on_eos_token_id, knn_memories, step=step)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
