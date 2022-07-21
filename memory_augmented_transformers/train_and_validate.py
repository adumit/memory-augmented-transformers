from tqdm import tqdm
import datetime

import mlflow
import torch


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


def run_epoch(model, optimizer, knn_memories, train_dataloader, epoch_ind, device):
    
    total_train_loss = 0.
    steps_taken = 0
    typical_batch_size = None
    
    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        
        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(
            b_input_ids,
            labels=b_labels, 
            attention_mask = b_masks,
            token_type_ids=None,
            knn_memories=knn_memories,
            step=step+(epoch_ind * len(train_dataloader))
        )

        loss = outputs[0]

        batch_loss = loss.item()
        total_train_loss += batch_loss
        mlflow.log_metric("Train loss", float(loss.detach().cpu().data), step=step)

        loss.backward()
        if model.transformer.g_per_head:
          for i, layer_g in enumerate(model.transformer.gs):
            layer_g_data = layer_g.data.cpu()
            layer_g_grad_data = layer_g.grad.data.cpu()
            layer = model.transformer.memory_layer_inds[i]
            gs = {f"g-{layer}-{head}": g for head, g in enumerate(layer_g_data)}
            g_grads = {f"g-grad-{i}": float(g_grad) for i,g_grad in enumerate(layer_g_grad_data)}
        else:
          gs = {f"g-{i}": float(g.data.cpu()) for i,g in enumerate(model.transformer.gs)}
          g_grads = {f"g-grad-{i}": float(g.grad.data.cpu()) for i,g in enumerate(model.transformer.gs)}
        mlflow.log_metrics(gs, step=step+(epoch_ind * len(train_dataloader)))
        mlflow.log_metrics(g_grads, step=step+(epoch_ind * len(train_dataloader)))

        optimizer.step()
        steps_taken += 1

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / steps_taken
    mlflow.log_metric("Avg epoch loss", float(avg_train_loss), step=step+(epoch_ind * len(train_dataloader)))


def run_validation(model, valid_dataloader, knn_memories, device, epoch=-1):
    print("Running Validation...")

    model.eval()

    total_eval_loss = 0
    steps_taken = 0
    typical_batch_size = None
    
    # Evaluate data for one epoch
    for batch in valid_dataloader:
        
        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)
        
        if typical_batch_size is None:
            typical_batch_size = b_input_ids.shape[0]
            
        if b_input_ids.shape[0] != typical_batch_size:
            continue

        with torch.no_grad():        

            outputs  = model(
                b_input_ids,
                attention_mask = b_masks,
                labels=b_labels,
                knn_memories=knn_memories)

            loss = outputs[0]  

        batch_loss = loss.item()
        total_eval_loss += batch_loss
        steps_taken += 1

    avg_val_loss = total_eval_loss / steps_taken

    mlflow.log_metric("Validation loss", float(avg_val_loss), step=epoch+1)

    return
