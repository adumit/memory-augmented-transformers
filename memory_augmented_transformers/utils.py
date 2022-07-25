from typing import *

from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from memory_augmented_transformers.gpt2_with_memory.lm_model import GPT2LMHeadModelWithMemory
from memory_augmented_transformers.gpt2_with_memory.model import GPT2WithMemory

def get_gpt_memory_model_and_tokenizer(
    model_id,
    device,
    mem_layer_inds, 
    num_mems, 
    max_mems,
    use_sigmoid_for_g,
    apply_linear_g,
    use_tanh_for_g=False,
    use_pass_through_knns=False,
    use_agg_before_layer=True,
    use_knn_mems_per_head=False,
    do_not_mem_grad_through_gpt_layers=False,
    g_per_head=False,
    normalize_qk=False
):

    base_model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

    memory_model = GPT2WithMemory(
        base_model.transformer, 
        memory_layer_inds=mem_layer_inds, 
        num_mems_retrieved=num_mems, 
        allow_body_finetuning=False,
        clear_memories_on_eos_token_id=tokenizer.eos_token_id,
        max_mems=max_mems,
        use_sigmoid_for_g=use_sigmoid_for_g,
        apply_linear_g=apply_linear_g,
        use_tanh_for_g=use_tanh_for_g,
        use_pass_through_knns=use_pass_through_knns,
        use_agg_before_layer=use_agg_before_layer,
        use_knn_mems_per_head=use_knn_mems_per_head,
        do_not_mem_grad_through_gpt_layers=do_not_mem_grad_through_gpt_layers,
        g_per_head=g_per_head,
        normalize_qk=normalize_qk
    )

    lm_head_model = GPT2LMHeadModel.from_pretrained(model_id)

    model = GPT2LMHeadModelWithMemory(
        config=base_model.config,
        transformer_with_memory=memory_model,
        lm_head_weight=lm_head_model.lm_head.weight
    )
    model = model.to(device)
    return model, tokenizer