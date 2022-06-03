from tqdm import tqdm

import mlflow
import torch


def run_epoch(model, optimizer, knn_memories, train_dataloader, epoch_ind, device='cuda'):
    
    total_train_loss = 0.
    steps_taken = 0
    typical_batch_size = None
    
    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        
        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)
        
        if typical_batch_size is None:
            typical_batch_size = b_input_ids.shape[0]

        model.zero_grad()
        
        if b_input_ids.shape[0] != typical_batch_size:
            continue
        
        outputs = model(
            b_input_ids,
            labels=b_labels, 
            attention_mask = b_masks,
            token_type_ids=None,
            knn_memories=knn_memories
        )

        loss = outputs[0] 

        batch_loss = loss.item()
        total_train_loss += batch_loss
        mlflow.log_metric("Train loss", float(loss.detach().cpu().data), step=step)

        loss.backward()

        optimizer.step()
        steps_taken += 1

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / steps_taken
    mlflow.log_metric("Avg epoch loss", float(avg_train_loss), step=step+(epoch_ind * len(train_dataloader)))


def run_validation(model, valid_dataloader, knn_memories, epoch=-1, device='cuda'):
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