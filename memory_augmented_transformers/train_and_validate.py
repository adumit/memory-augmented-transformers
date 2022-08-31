from tqdm import tqdm
import datetime

import mlflow
import torch


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


def run_epoch(model, optimizer, knn_memories, train_dataloader, epoch_ind, track_loss_per_doc=False):
    
    loss_per_doc = defaultdict(lambda: 0)
    total_train_loss = 0.
    steps_taken = 0
    typical_batch_size = None
    
    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        
        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_attn_masks = batch[1].to(device)
        b_loss_masks = batch[2].to(device)
        
        # if typical_batch_size is None:
        #     typical_batch_size = b_input_ids.shape[0]

        # model.zero_grad()
        optimizer.zero_grad(set_to_none=True)
        
        # if b_input_ids.shape[0] != typical_batch_size:
        #     continue
        
        outputs, unreduced_loss = model(
            b_input_ids,
            labels=b_labels, 
            attention_mask = b_attn_masks,
            loss_mask = b_loss_masks,
            token_type_ids=None,
            knn_memories=knn_memories,
            step=step+(epoch_ind * len(train_dataloader))
        )

        loss = outputs[0]
        if track_loss_per_doc:
            batch_size = b_input_ids.shape[0]
            loss_per_item_in_batch = unreduced_loss.view(batch_size, -1).detach().cpu().mean(dim=0)
            for i, item_loss in enumerate(loss_per_item_in_batch):
                dataset_ind = step * batch_size + i
                if dataset_ind not in train_dataloader.dataset.item_index_iterator:
                    continue
                item_index = train_dataloader.dataset.item_index_iterator[dataset_ind]
                doc_index = train_dataloader.dataset.item_index_to_document_index[item_index]
                loss_per_doc[doc_index] += item_loss

        batch_loss = loss.item()
        total_train_loss += batch_loss

        loss.backward()
        if model.transformer.g_per_head:
            for i, layer_g in enumerate(model.transformer.gs):
                layer_g_data = layer_g.data.cpu()
                layer = model.transformer.memory_layer_inds[i]
                gs = {f"g-{layer}-{head}": g for head, g in enumerate(layer_g_data)}
        else:
            gs = {f"g-{i}": float(g.data.cpu()) for i,g in enumerate(model.transformer.gs)}

        if model.transformer.knn_attns and hasattr(model.transformer.knn_attns[0], "scale"):
            for i, knn_attn in enumerate(model.transformer.knn_attns):
                scale_val = knn_attn.scale.data.cpu()
                for j, sval in enumerate(scale_val):
                    gs[f"scale_param-{i}-{j}"] = float(sval)
                    
        gs["Train loss"] = float(loss.detach().cpu().data)
        mlflow.log_metrics(gs, step=step+(epoch_ind * len(train_dataloader)))
        # mlflow.log_metrics(g_grads, step=step+(epoch_ind * len(train_dataloader)))

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
