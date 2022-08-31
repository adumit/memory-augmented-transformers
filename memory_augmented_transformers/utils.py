from typing import *

from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from memory_augmented_transformers.gpt2_with_memory.lm_model import GPT2LMHeadModelWithMemory
from memory_augmented_transformers.gpt2_with_memory.model import GPT2WithMemory

def get_gpt_memory_model_and_tokenizer(
    model_id,
    mem_layer_inds, 
    num_mems, 
    max_mems,
    use_sigmoid_for_g,
    apply_linear_g,
    device,
    use_tanh_for_g=False,
    use_softmax_over_localdistant_layer=False,
    use_knn_mems_per_head=False,
    g_per_head=False,
    normalize_qk=False,
    return_unreduced_loss=False,
    normalize_query_for_attn_mult=False,
    include_scale_parameter=False,
    create_memory_gpt_layer_copy=False,
    allow_body_finetuning=False
):

    base_model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

    memory_model = GPT2WithMemory(
        base_model.transformer, 
        memory_layer_inds=mem_layer_inds, 
        num_mems_retrieved=num_mems, 
        allow_body_finetuning=allow_body_finetuning,
        clear_memories_on_eos_token_id=tokenizer.eos_token_id,
        max_mems=max_mems,
        use_sigmoid_for_g=use_sigmoid_for_g,
        apply_linear_g=apply_linear_g,
        use_tanh_for_g=use_tanh_for_g,
        use_knn_mems_per_head=use_knn_mems_per_head,
        g_per_head=g_per_head,
        normalize_qk=normalize_qk,
        use_softmax_over_localdistant_layer=use_softmax_over_localdistant_layer,
        normalize_query_for_attn_mult=normalize_query_for_attn_mult,
        include_scale_parameter=include_scale_parameter,
        create_memory_gpt_layer_copy=create_memory_gpt_layer_copy,
    )

    lm_head_model = GPT2LMHeadModel.from_pretrained(model_id)

    model = GPT2LMHeadModelWithMemory(
        config=base_model.config,
        transformer_with_memory=memory_model,
        lm_head_weight=lm_head_model.lm_head.weight,
        return_unreduced_loss=return_unreduced_loss
    )
    model = model.to(device)
    return model, tokenizer