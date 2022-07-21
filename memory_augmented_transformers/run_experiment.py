import os

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from memory_augmented_transformers.utils import get_gpt_memory_model_and_tokenizer
from memory_augmented_transformers.dataset.worldbank_dataset import get_worldbank_project_document_dataset
from memory_augmented_transformers.train_and_validate import run_epoch, run_validation


def run_an_experiment(
    run_name,
    device,
    epochs=2,
    lr=5e-4,
    warmup_steps=1e2,
    epsilon=1e-8,
    batch_size=2,
    num_memories=3,
    maximum_memories=16384,
    layers_to_insert_memories=(3,),
    dataset_to_use="PG-19",
    finetune_body=False,
    finetune_head=False,
    unfreeze_head_after_n_epochs=100,
    unfreeze_knn_attns_after_n_epochs=100,
    learning_rate_part_2=None,
    head_mem_body_lrs=None,
    use_sigmoid_for_g=False,
    apply_linear_g=True,
    mem_transformer_dim_head=64,
    mem_transformer_heads=12,
    apply_gated_xattn=False,
    use_tanh_for_g=False,
    use_pass_through_knns=False,
    use_agg_before_layer=True,
    use_knn_mems_per_head=False,
    do_not_mem_grad_through_gpt_layers=False,
    g_per_head=False,
    normalize_qk=False
):

    mlflow.end_run()
    mlflow.end_run()

    rand_state = 1
    torch.manual_seed(rand_state)

    # MLFlow will log to a local runs directory (./runs) if the environment variables aren't set for DagsHub
    if "MLFLOW_TRACKING_USERNAME" in os.environ:
        mlflow.set_tracking_uri(f'https://dagshub.com/' + os.environ['MLFLOW_TRACKING_USERNAME'] + '/' + os.environ['MLFLOW_TRACKING_PROJECTNAME'] + '.mlflow')
    mlflow.set_experiment("Memory-augmented-transformer-2022-06-07")
    mlflow.start_run(run_name=run_name)
    
    # TODO: Switch here once other models are implemented
    model_id = "distilgpt2"
    epochs = epochs
    learning_rate = lr
    warmup_steps = warmup_steps
    epsilon = epsilon
    batch_size = batch_size
    num_memories = num_memories

    param_dict = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "epsilon": epsilon,
        "batch_size": batch_size,
        "num_memories": num_memories,
        "rand_state": rand_state,
        "dataset_to_use": dataset_to_use,
        "layers_to_insert_memories": layers_to_insert_memories,
        "finetune_body": finetune_body,
        "finetune_head": finetune_head,
        "unfreeze_head_after_n_epochs": unfreeze_head_after_n_epochs,
        "learning_rate_part_2": learning_rate_part_2,
        "maximum_memories": maximum_memories,
        "use_sigmoid_for_g": use_sigmoid_for_g,
        "apply_linear_g": apply_linear_g,
        "dim_head": mem_transformer_dim_head,
        "heads": mem_transformer_heads,
        "apply_gated_xattn": apply_gated_xattn,
        "use_tanh_for_g": use_tanh_for_g,
        "use_pass_through_knns": use_pass_through_knns,
        "use_agg_before_layer": use_agg_before_layer,
        "use_knn_mems_per_head": use_knn_mems_per_head,
        "do_not_mem_grad_through_gpt_layers": do_not_mem_grad_through_gpt_layers,
        "g_per_head": g_per_head,
        "normalize_qk": normalize_qk,
        "dataset": dataset_to_use
    }
    if head_mem_body_lrs:
      param_dict["learning_rate"] = head_mem_body_lrs
      param_dict["head_lr"] = head_mem_body_lrs[0]
      param_dict["mem_lr"] = head_mem_body_lrs[1]
      param_dict["body_lr"] = head_mem_body_lrs[2]

    mlflow.log_params(param_dict)


    model, tokenizer = get_gpt_memory_model_and_tokenizer(
        model_id,
        device,
        mem_layer_inds=layers_to_insert_memories,
        num_mems=num_memories,
        max_mems=maximum_memories,
        use_sigmoid_for_g=use_sigmoid_for_g,
        apply_linear_g=apply_linear_g,
        dim_head=mem_transformer_dim_head,
        heads=mem_transformer_heads,
        apply_gated_xattn=apply_gated_xattn,
        use_tanh_for_g=use_tanh_for_g,
        use_pass_through_knns=use_pass_through_knns,
        use_agg_before_layer=use_agg_before_layer,
        use_knn_mems_per_head=use_knn_mems_per_head,
        do_not_mem_grad_through_gpt_layers=do_not_mem_grad_through_gpt_layers,
        g_per_head=g_per_head,
        normalize_qk=normalize_qk
    )
    
    if finetune_head:
        model.unfreeze_head()

    if finetune_body:
      model.transformer.unfreeze_body()
    
    # TODO: Implement PG-19 dataset gathering for github
    # if "PG-19" in dataset_to_use:
    #   perc_to_use = int(dataset_to_use.split("-")[-1][:-1]) / 100
    #   train_dataset, valid_dataset = get_PG_19_sample_datasets(tokenizer, batch_size, sample_size=perc_to_use)
    if "worldbank" in dataset_to_use:
        perc = int(dataset_to_use.split("_")[-1]) / 100
        train_dataset, valid_dataset = get_worldbank_project_document_dataset(tokenizer, batch_size, sample_perc=perc)
    else:
        raise RuntimeError("Unsupported dataset")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle=False  # Do not shuffle because we need to see the book in order
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False  # Do not shuffle because we need to see the book in order
    )

    model.train()

    if head_mem_body_lrs:
      # Handle co-training the memory and head layers
      if head_mem_body_lrs[2] is not None:
        model.transformer.unfreeze_body()
      if head_mem_body_lrs[0] is not None:
        model.unfreeze_head()

      param_map = [
          {'params': model.transformer.get_knn_parameters(), 'lr': head_mem_body_lrs[1]}
      ]
      if head_mem_body_lrs[0] is not None:
        param_map.append({'params': model.lm_head.parameters(), 'lr': head_mem_body_lrs[0]})
      if head_mem_body_lrs[2] is not None:
        param_map.append({'params': model.transformer.get_body_parameters(), 'lr': head_mem_body_lrs[2]})

      optimizer = AdamW(
          param_map,
          lr=head_mem_body_lrs[1],
          eps=epsilon
      )
    else:
      optimizer = AdamW(
          model.parameters(),
          lr = learning_rate,
          eps = epsilon
      )

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of trainable parameters:", params)
    
    run_validation(
        model=model,
        valid_dataloader=valid_dataloader,
        knn_memories=None,
        epoch=-1
    )
    
    with model.transformer.knn_memories_context(batch_size = train_dataloader.batch_size) as knn_memories:
        for epoch in range(epochs):
            if epoch + 1 > unfreeze_head_after_n_epochs:
                model.unfreeze_head()
                optimizer = AdamW(
                    model.parameters(),
                    lr = learning_rate_part_2,
                    eps = epsilon
                )
            if epoch + 1 > unfreeze_knn_attns_after_n_epochs:
                model.freeze_head()
                optimizer = AdamW(
                    model.parameters(),
                    lr = learning_rate_part_2,
                    eps = epsilon
                )
                
            run_epoch(
                model=model, 
                optimizer=optimizer, 
                knn_memories=knn_memories, 
                train_dataloader=train_dataloader, 
                epoch_ind=epoch
            )
            run_validation(
                model=model,
                valid_dataloader=valid_dataloader,
                knn_memories=knn_memories,
                epoch=epoch
            )
    mlflow.end_run()
    mlflow.end_run()
